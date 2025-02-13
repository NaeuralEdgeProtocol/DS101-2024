
import torch
import torch.nn as nn
# Parameters for the model
parameters = {
    'max_len': 60,
    'num_lstm_layers': 2,
    'units_l0': 107,
    'dropout_lstm_l0': True,
    'dropout_rate_lstm_l0': 0.14358991845052868,
    'units_l1': 41,
    'dropout_lstm_l1': False,
    'num_dense_layers': 2,
    'units_l_dense0': 94,
    'dropout_dense_l0': True,
    'dropout_dense_rate_l0': 0.2819323238505278,
    'units_l_dense1': 95,
    'dropout_dense_l1': True,
    'dropout_dense_rate_l1': 0.16012611463695886,
    'optimizer': 'adam',
    'epochs': 3,
}

class BLSTMClassifier(nn.Module):
    def __init__(self, params):
        super(BLSTMClassifier, self).__init__()
        self.max_len = params['max_len']
        
        # First LSTM layer (bidirectional)
        self.lstm0 = nn.LSTM(
            input_size=300,
            hidden_size=params['units_l0'],
            num_layers=1,       # we use one layer here; we build a second LSTM separately
            batch_first=True,
            bidirectional=True
        )
        self.use_dropout_lstm0 = params['dropout_lstm_l0']
        if self.use_dropout_lstm0:
            self.dropout_lstm0 = nn.Dropout(params['dropout_rate_lstm_l0'])
        
        
        self.lstm1 = nn.LSTM(
            input_size=params['units_l0'] * 2,
            hidden_size=params['units_l1'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        

        final_feature_size = params['units_l1'] * 2  # 41*2 = 82

        # Dense layers
        self.fc0 = nn.Linear(final_feature_size, params['units_l_dense0'])
        self.relu = nn.ReLU()
        self.use_dropout_dense0 = params['dropout_dense_l0']
        if self.use_dropout_dense0:
            self.dropout_dense0 = nn.Dropout(params['dropout_dense_rate_l0'])
            
        self.fc1 = nn.Linear(params['units_l_dense0'], params['units_l_dense1'])
        self.use_dropout_dense1 = params['dropout_dense_l1']
        if self.use_dropout_dense1:
            self.dropout_dense1 = nn.Dropout(params['dropout_dense_rate_l1'])
            
        # Final output layer: one neuron for binary classification.
        self.out = nn.Linear(params['units_l_dense1'], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, max_len, 300)
        # Pass through first LSTM layer
        out_lstm0, _ = self.lstm0(x)  # out_lstm0 shape: (batch, max_len, units_l0*2)
        if self.use_dropout_lstm0:
            out_lstm0 = self.dropout_lstm0(out_lstm0)
            
        # Pass through second LSTM layer
        out_lstm1, (h_n, c_n) = self.lstm1(out_lstm0)
        # h_n shape: (num_directions, batch, units_l1)
        # Concatenate the forward and backward hidden states from the second LSTM.
        h_forward = h_n[0]  # shape: (batch, units_l1)
        h_backward = h_n[1]  # shape: (batch, units_l1)
        h_final = torch.cat((h_forward, h_backward), dim=1)  # shape: (batch, 2*units_l1)
        
        # Dense layers
        x_dense = self.fc0(h_final)
        x_dense = self.relu(x_dense)
        if self.use_dropout_dense0:
            x_dense = self.dropout_dense0(x_dense)
            
        x_dense = self.fc1(x_dense)
        x_dense = self.relu(x_dense)
        if self.use_dropout_dense1:
            x_dense = self.dropout_dense1(x_dense)
            
        logits = self.out(x_dense)
        probs = self.sigmoid(logits)
        return probs

    def predict_single(self, example):
        """
        Predict the class label (0 or 1) for a single example.
        Input:
            example: numpy array or torch tensor of shape (max_len, 300)
        Returns:
            predicted label: 0 or 1
        """
        self.eval()
        if isinstance(example, np.ndarray):
            example = torch.tensor(example, dtype=torch.float32)
        # Add batch dimension: (1, max_len, 300)
        example = example.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = self.forward(example)
        # Threshold at 0.5
        label = (prob.item() >= 0.5)
        return int(label)