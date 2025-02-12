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



# loaded_model, loaded_scaler = load_drone_flight_model(model_path='best_drone_flight_model.pth', scaler_path='drone_flight_scaler.pkl')
loaded_model, loaded_scaler = load_drone_flight_model(model_path='bilstm/best_drone_flight_model.pth', scaler_path='bilstm/drone_flight_scaler.pkl')

# Example prediction
prediction = predict_drone_flight(
    loaded_model, 
    loaded_scaler, 
    "flight_Data_27.csv"
)
print("Prediction:", prediction)