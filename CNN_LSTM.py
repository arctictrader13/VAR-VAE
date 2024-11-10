import torch
import torch.nn as nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class CNN_LSTM(nn.Module):
    def __init__(self, num_features, encoding_dim, window_size):
        super(CNN_LSTM, self).__init__()
        
        # Encoder CNN layers with BatchNorm and Dropout
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)  # BatchNorm for the first conv layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)  # Dropout with probability 0.3
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)  # BatchNorm for the second conv layer
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)  # BatchNorm for the third conv layer
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)  # Dropout with probability 0.3

        # Encoder linear layers with Dropout
        self.fc1 = nn.Linear(832, 32)
        self.bn4 = nn.BatchNorm1d(32)  # Corrected: BatchNorm for the first fully connected layer to match 32 features
        self.dropout3 = nn.Dropout(p=0.3)  # Dropout with probability 0.3

        # LSTM layer to replace the variational approach
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(32)
        self.dropout_lstm = nn.Dropout(p=0.3)  # Dropout with probability 0.3
        
        # Linear layers after LSTM
        self.fc2 = nn.Linear(32, 32)
        self.bn5 = nn.BatchNorm1d(32)  # Corrected: BatchNorm for the second fully connected layer to match 64 features
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.3)  # Dropout with probability 0.3
        
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.dropout2(x)

        # Flatten and apply linear layers
        x = x.view(x.shape[0], -1)  # Flatten for fully connected layer
        x = self.dropout3(self.bn4(self.fc1(x)))

        # LSTM expects a 3D input (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # Add a sequence length dimension for LSTM (batch_size, 1, input_size)
        x, (hn, cn) = self.lstm(x)  # Unpack the LSTM output
        
        # Apply BatchNorm and Dropout to the LSTM output
        x = x[:, -1, :]  # Take the last output of the LSTM
        x = self.bn_lstm(x)  # BatchNorm applied to the features
        x = self.dropout_lstm(x)  # Apply dropout to LSTM output
        
        x = self.relu4(self.bn5(self.fc2(x)))
        x = self.dropout4(x)
        
        x = self.fc3(x)
        
        return x