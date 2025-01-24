# Task B - BloodMNIST Multi-class Classification
# Model implementation for blood cell classification
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""
BloodMNIST Multi-class Classification Model

Architecture Justification:
1. Deeper CNN with batch normalization - handles more complex features
2. More filters - captures diverse blood cell characteristics
3. Additional FC layer - increased capacity for multi-class
4. Cross-entropy loss - standard for multi-class classification
5. Larger batch size - better gradient estimates for multiple classes
"""


class BloodCNNModel(nn.Module):
    def __init__(self):
        super(BloodCNNModel, self).__init__()
        # More complex CNN for multi-class classification
        self.conv_layers = nn.Sequential(
            # First conv block with batch norm
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Adjusted input channels to 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Deeper FC layers for multi-class
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  # 8 classes output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)  # Add this line to check the shape
        return self.fc_layers(x)




if __name__ == "__main__":
    from main import DataManager  # Import DataManager from main.py
    data_manager = DataManager(data_dir="../Datasets")  # Pass parent directory as data_dir
    blood_data = data_manager.load_blood_data()

