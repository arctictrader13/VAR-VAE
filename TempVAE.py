import torch
import torch.nn as nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class TempVAE(nn.Module):
    def __init__(self, num_features, encoding_dim, window_size):
        super(TempVAE, self).__init__()
        
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
        self.bn4 = nn.BatchNorm1d(1)  # BatchNorm for the first fully connected layer
        self.dropout3 = nn.Dropout(p=0.3)  # Dropout with probability 0.3

        # Latent space
        self.mean = nn.Linear(32, 32)
        self.log_var = nn.Linear(32, 32)
        
        self.fc2 = nn.Linear(32, 32)
        self.bn5 = nn.BatchNorm1d(1)  # BatchNorm for the second fully connected layer
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
        x = x.view(x.shape[0], 1, 832)
        x = self.dropout3(self.bn4(self.fc1(x)))
        
        mean = self.mean(x)
        std_ = torch.exp(0.5 * self.log_var(x))

        eps = torch.randn_like(std_)
        
        z = mean + eps * std_
        
        ###########################################################
        # VAR part
        z_hat = self.fc2(z)
        
        ###########################################################
    
        x = self.relu4(self.bn5(z_hat))
        x = self.dropout4(x)
        
        x = self.fc3(x)
        
        return x, mean, std_, z, z_hat