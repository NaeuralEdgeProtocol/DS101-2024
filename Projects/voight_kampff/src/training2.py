#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Load and preprocess the data
def load_data(bots_path, humans_path):
    print(f"Loading bot data from {bots_path}")
    with open(bots_path, 'r', encoding='utf-8') as f:
        bots_data = json.load(f)
    print(f"Found {len(bots_data)} bot entries")
    
    print(f"Loading human data from {humans_path}")
    with open(humans_path, 'r', encoding='utf-8') as f:
        humans_data = json.load(f)
    print(f"Found {len(humans_data)} human entries")
    
    # Create dictionaries with labels
    labeled_data = {}
    
    # Process bot data
    for username, data in bots_data.items():
        if isinstance(data, dict):  # Verify data structure
            labeled_data[username] = {'data': data, 'label': 1}  # 1 for bots
    
    # Process human data
    for username, data in humans_data.items():
        if isinstance(data, dict):  # Verify data structure
            labeled_data[username] = {'data': data, 'label': 0}  # 0 for humans
    
    print(f"Successfully processed {len(labeled_data)} total entries")
    print(f"Number of bots: {sum(1 for x in labeled_data.values() if x['label'] == 1)}")
    print(f"Number of humans: {sum(1 for x in labeled_data.values() if x['label'] == 0)}")
    
    return labeled_data

def preprocess_user_data(user_data):
    # Combine all text data for the user
    text_data = []
    
    # Add comments
    if 'Latest Comments' in user_data:
        comments = [comment.get('Comment', '') for comment in user_data['Latest Comments'] if isinstance(comment, dict)]
        comments = [c for c in comments if c]  # Remove empty comments
        text_data.extend(comments[:5])  # Use up to 5 latest comments
    
    # Add post titles
    if 'Latest Posts' in user_data:
        titles = [post.get('Title', '') for post in user_data['Latest Posts'] if isinstance(post, dict)]
        titles = [t for t in titles if t]  # Remove empty titles
        text_data.extend(titles[:5])  # Use up to 5 latest post titles
    
    # Combine all text with [SEP] token
    combined_text = ' [SEP] '.join(text_data) if text_data else ''
    
    # Extract numerical features with default values
    default_numerical_features = [0.0] * 6  # Default values for all features
    
    if 'User Metrics' in user_data and isinstance(user_data['User Metrics'], dict):
        metrics = user_data['User Metrics']
        try:
            numerical_features = [
                float(metrics.get('Average Posts Per Day', 0)),
                float(metrics.get('Average Comments Per Day', 0)),
                float(metrics.get('Average Post Time Gap (hrs)', 0)),
                float(metrics.get('Average Comment Time Gap (hrs)', 0)),
                float(metrics.get('Average Comment Length', 0)),
                float(metrics.get('Upvote/Downvote Ratio (Posts)', 0))
            ]
        except (ValueError, TypeError):
            numerical_features = default_numerical_features
    else:
        numerical_features = default_numerical_features
    
    # Ensure all features are finite
    numerical_features = [0.0 if not np.isfinite(x) else float(x) for x in numerical_features]
    
    return combined_text, torch.tensor(numerical_features, dtype=torch.float)

# Custom Dataset
class RedditUserDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.usernames = list(data.keys())
        
    def __len__(self):
        return len(self.usernames)
    
    def __getitem__(self, idx):
        username = self.usernames[idx]
        user_entry = self.data[username]
        user_data = user_entry['data']
        label = user_entry['label']
        
        # Process text and numerical features
        text, numerical_features = preprocess_user_data(user_data)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to tensors
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'numerical_features': numerical_features,
            'label': torch.tensor(label, dtype=torch.float)
        }

# Neural Network Model
class BotDetectionModel(nn.Module):
    def __init__(self, bert_model, numerical_features=6, hidden_size=256):
        super().__init__()
        self.bert = bert_model
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.bert_dropout = nn.Dropout(0.1)
        
        # Neural network for combined features
        self.classifier = nn.Sequential(
            nn.Linear(768 + numerical_features, hidden_size),  # 768 is BERT's hidden size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask, numerical_features):
        # Get BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        pooled_output = self.bert_dropout(pooled_output)
        
        # Concatenate BERT embeddings with numerical features
        combined_features = torch.cat([pooled_output, numerical_features], dim=1)
        
        # Pass through classifier
        return self.classifier(combined_features)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=2e-5, patience=5, min_improvement=0.001):
    """
    Train the model with early stopping
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs (int): Maximum number of epochs to train
        learning_rate (float): Learning rate for optimizer
        patience (int): Number of epochs to wait for improvement before early stopping
        min_improvement (float): Minimum improvement in validation accuracy to reset patience counter
    """
    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print("Starting training with early stopping:")
    print(f"Patience: {patience} epochs")
    print(f"Minimum improvement threshold: {min_improvement}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Training accuracy: {train_accuracy:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numerical_features = batch['numerical_features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, numerical_features)
                val_loss += criterion(outputs.squeeze(), labels).item()
                
                predictions = (outputs.squeeze() > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        
        # Store metrics
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Validation loss: {avg_val_loss:.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        
        # Check for improvement
        if val_accuracy > best_val_acc + min_improvement:
            print(f"Validation accuracy improved by {val_accuracy - best_val_acc:.4f}")
            best_val_acc = val_accuracy
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No significant improvement. Patience counter: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
    
    # Load best model and save training history
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nTraining completed. Loading best model state.")
        print(f"Final best validation accuracy: {best_val_acc:.4f}")
    
    # Plot and save training curves
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curves_{timestamp}.png')
    plt.close()
    
    # Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'accuracy_curves_{timestamp}.png')
    plt.close()
    
    # Save metrics to file
    metrics = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies,
        'best_val_acc': float(best_val_acc),  # Convert to float for JSON serialization
        'epochs_completed': len(train_losses),
        'stopped_early': len(train_losses) < num_epochs
    }
    with open(f'training_metrics_{timestamp}.json', 'w') as f:
        json.dump(metrics, f)
    
    return model

# Main execution
def main():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Load data
    data = load_data('../data/bots_data.json', '../data/humans_data.json')
    
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)  # Move BERT to GPU
    
    # Create dataset
    dataset = RedditUserDataset(data, tokenizer)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = BotDetectionModel(bert_model).to(device)
    
    # Train model with early stopping
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,           # Maximum number of epochs
        learning_rate=2e-5,      # Learning rate
        patience=5,              # Early stopping patience
        min_improvement=0.001    # Minimum improvement threshold (0.1%)
    )
    
    # Save model
    torch.save(model.state_dict(), 'bot_detection_model.pth')

if __name__ == "__main__":
    main()