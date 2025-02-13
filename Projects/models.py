
# BiLSSTM model for drone flight data classification
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F

class VariableLengthDroneFlightDataset(Dataset):
    """
    PyTorch Dataset for drone flight data with variable sequence lengths
    """
    def __init__(self, file_list, scaler=None, is_training=True):
        self.file_list = file_list
        self.is_training = is_training
        self.scaler = scaler if scaler is not None else StandardScaler()
        
        # Determine if files are faulty based on path
        self.labels = torch.tensor(['Faulty' in f for f in self.file_list], dtype=torch.float32)
        
        # Fit scaler on training data
        if is_training and scaler is None:
            self._fit_scaler()

    def _fit_scaler(self):
        print("Fitting scaler on data subset...")
        sample_data = []
        subset_size = min(500, len(self.file_list))
        subset_files = np.random.choice(self.file_list, subset_size, replace=False)
        
        for file in tqdm(subset_files, desc="Fitting Scaler"):
            df = pd.read_csv(file)
            sample_data.append(df.values)
        
        sample_data = np.vstack(sample_data)
        self.scaler.fit(sample_data)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        df = pd.read_csv(file)
        data = df.values
        
        # Scale the data
        data = self.scaler.transform(data)
        
        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        
        return data, self.labels[idx]

def collate_variable_length(batch):
    """
    Modified collate function with memory constraints
    """
    # Sort by length for packed sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    
    # Get lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Limit maximum sequence length to prevent memory issues
    max_len = min(lengths.max().item(), 10000)  # Set reasonable maximum length
    sequences = [seq[:max_len] for seq in sequences]
    
    # Pad sequences
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
    
    # Ensure tensors are contiguous
    padded_sequences = padded_sequences.contiguous()
    lengths = lengths.clamp(max=max_len).contiguous()
    labels = torch.tensor(labels, dtype=torch.float32).contiguous()
    
    return padded_sequences, lengths, labels

class VariableLengthLSTMModel(nn.Module):
    def __init__(self, input_features, hidden_size=128, num_layers=3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input dropout
        self.input_dropout = nn.Dropout(0.2)
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_features, 
            hidden_size, 
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            hidden_size * 2,
            hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout_fc1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn_fc2 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout_fc2 = nn.Dropout(0.4)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size // 4, 1)
    def forward(self, x, lengths):
        # Apply input dropout
        x = self.input_dropout(x)
        
        # Pack sequence for first LSTM
        packed_input = rnn_utils.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # First LSTM layer
        packed_output1, _ = self.lstm1(packed_input)
        output1, _ = rnn_utils.pad_packed_sequence(packed_output1, batch_first=True)
        
        # Apply batch norm and dropout
        output1 = self.bn1(output1.transpose(1, 2)).transpose(1, 2)
        output1 = self.dropout1(output1)
        
        # Pack sequence for second LSTM
        packed_output1_packed = rnn_utils.pack_padded_sequence(
            output1, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Second LSTM layer
        packed_output2, _ = self.lstm2(packed_output1_packed)
        output2, _ = rnn_utils.pad_packed_sequence(packed_output2, batch_first=True)
        
        # Apply batch norm and dropout
        output2 = self.bn2(output2.transpose(1, 2)).transpose(1, 2)
        output2 = self.dropout2(output2)
        
        # Attention mechanism
        attention_weights = self.attention(output2)
        context_vector = torch.sum(attention_weights * output2, dim=1)
        
        # Fully connected layers with residual connections
        out = self.fc1(context_vector)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout_fc1(out)
        
        residual = out
        
        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = F.relu(out)
        out = self.dropout_fc2(out)
        
        # Add residual connection if dimensions match
        if residual.shape[1] == out.shape[1]:
            out = out + residual
        
        # Output layer
        out = self.fc_out(out)
        out = torch.sigmoid(out)
        
        return out

def train_variable_length_model(
    normal_path, 
    faulty_path, 
    batch_size=16,
    epochs=30,
    learning_rate=0.0001,
    validation_split=0.2,
    patience=10,
    weight_decay=0.001,
    best_model_save_path='best_drone_flight_model.pth',
    latest_model_save_path='latest_drone_flight_model.pth',
    scaler_save_path='drone_flight_scaler.pkl'
):
    # Get file lists
    normal_files = glob.glob(f"{normal_path}/flight_Data_*.csv")
    faulty_files = glob.glob(f"{faulty_path}/flight_Data_*.csv")
    all_files = normal_files + faulty_files
    
    print(f"Total files found: {len(all_files)}")
    print(f"Normal files: {len(normal_files)}")
    print(f"Faulty files: {len(faulty_files)}")
    
    # Create dataset
    full_dataset = VariableLengthDroneFlightDataset(all_files, is_training=True)
    
    # Initialize k-fold cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Get sample batch to determine input features
    temp_loader = DataLoader(full_dataset, batch_size=1, collate_fn=collate_variable_length)
    sample_batch = next(iter(temp_loader))
    input_features = sample_batch[0].shape[2]
    print(f"\nInput Features: {input_features}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Initialize metrics storage
    all_fold_metrics = []
    best_val_loss = float('inf')
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize fold counter
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'\nFOLD {fold+1}/{k_folds}')
        print('-' * 50)
        
        # Create data samplers for current fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Create data loaders
        train_loader = DataLoader(
            full_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            collate_fn=collate_variable_length,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            full_dataset, 
            batch_size=batch_size, 
            sampler=val_sampler,
            collate_fn=collate_variable_length,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize model, criterion, optimizer
        model = VariableLengthLSTMModel(input_features).to(device)
        
        # Calculate pos_weight for imbalanced dataset
        pos_weight = torch.tensor([len(normal_files)/len(faulty_files)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Initialize metrics for current fold
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        epochs_no_improve = 0
        
        # Training loop for current fold
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Fold {fold+1}, Epoch {epoch+1} Training')
            for sequences, seq_lengths, labels in train_pbar:
                # Move data to device
                sequences = sequences.to(device)
                seq_lengths = seq_lengths.to(device)
                labels = labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(sequences, seq_lengths).squeeze()
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                
                # Update scheduler
                scheduler.step()
                
                # Compute accuracy
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                total_train_loss += loss.item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': train_correct/train_total
                })
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_true_labels = []
            
            val_pbar = tqdm(val_loader, desc=f'Fold {fold+1}, Epoch {epoch+1} Validation')
            with torch.no_grad():
                for sequences, seq_lengths, labels in val_pbar:
                    sequences = sequences.to(device)
                    seq_lengths = seq_lengths.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(sequences, seq_lengths).squeeze()
                    loss = criterion(outputs, labels)
                    
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    total_val_loss += loss.item()
                    
                    val_predictions.extend(predicted.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())
                    
                    val_pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': val_correct/val_total
                    })
            
            # Calculate epoch metrics
            train_loss = total_train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            val_loss = total_val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Print epoch results
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                # Save best model
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'input_features': input_features
                }, best_model_save_path)
                
                print(f'Best model saved with validation loss: {val_loss:.4f}')
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Store fold metrics
        all_fold_metrics.append({
            'fold': fold,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_val_loss': val_loss,
            'final_val_accuracy': val_accuracy
        })
        
        # Plot fold results
        plt.figure(figsize=(15, 5))
        
        # Loss Curve
        plt.subplot(131)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Loss Curve - Fold {fold+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy Curve
        plt.subplot(132)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title(f'Accuracy Curve - Fold {fold+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Confusion Matrix
        plt.subplot(133)
        cm = confusion_matrix(val_true_labels, val_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_fold_{fold+1}.png')
        plt.close()
    
    # Calculate and print average metrics across all folds
    avg_final_val_loss = np.mean([m['final_val_loss'] for m in all_fold_metrics])
    avg_final_val_acc = np.mean([m['final_val_accuracy'] for m in all_fold_metrics])
    
    print('\nCross-validation Results:')
    print(f'Average Validation Loss: {avg_final_val_loss:.4f}')
    print(f'Average Validation Accuracy: {avg_final_val_acc:.4f}')
    
    # Save final model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_features': input_features,
        'cross_val_metrics': all_fold_metrics
    }, latest_model_save_path)
    
    from joblib import dump
    dump(full_dataset.scaler, scaler_save_path)
    
    print(f"\nFinal model saved to {latest_model_save_path}")
    print(f"Scaler saved to {scaler_save_path}")
    
    return model, all_fold_metrics

def load_drone_flight_model(
    model_path='best_drone_flight_model.pth', 
    scaler_path='drone_flight_scaler.pkl'
):
    """
    Load a pre-trained drone flight classification model and its associated scaler
    
    Args:
        model_path (str): Path to the saved model checkpoint
        scaler_path (str): Path to the saved StandardScaler
    
    Returns:
        tuple: (loaded model, loaded scaler)
    """
    # Load scaler
    from joblib import load
    scaler = load(scaler_path)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create model with same architecture used during training
    model = VariableLengthLSTMModel(checkpoint['input_features'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model, scaler

def predict_drone_flight(
    model, 
    scaler, 
    flight_data_path, 
    threshold=0.5
):
    """
    Predict whether a drone flight is faulty or normal
    
    Args:
        model (VariableLengthLSTMModel): Trained LSTM model
        scaler (StandardScaler): Fitted StandardScaler
        flight_data_path (str): Path to the flight data CSV
        threshold (float): Classification threshold (default 0.5)
    
    Returns:
        dict: Prediction results including probability and classification
    """
    # Load flight data
    df = pd.read_csv(flight_data_path)
    data = df.values
    
    # Scale the data
    scaled_data = scaler.transform(data)
    
    # Convert to tensor
    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    seq_lengths = torch.tensor([len(data_tensor[0])])
    
    # Predict
    with torch.no_grad():
        output = model(data_tensor, seq_lengths).squeeze()
        probability = output.item()
        is_faulty = probability > threshold
    
    return {
        'probability': probability,
        'is_faulty': is_faulty,
        'threshold': threshold
    }

# Example usage
if __name__ == "__main__":
    # Training example
    model, metrics = train_variable_length_model(
    normal_path="CopyDS/Norm",
    faulty_path="CopyDS/Faults",
    batch_size=16,
    epochs=30,
    learning_rate=0.0001,
    patience=10,
    weight_decay=0.001
)
    # Loading and prediction example
    loaded_model, loaded_scaler = load_drone_flight_model()
    
    # Example prediction
    prediction = predict_drone_flight(
        loaded_model, 
        loaded_scaler, 
        "flight_Data_27.csv"
    )
    print("Prediction:", prediction)
