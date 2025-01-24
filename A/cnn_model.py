# Task A - BreastMNIST Binary Classification
# Model implementation for breast cancer detection
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""
BreastMNIST Binary Classification Model

Architecture Justification:
1. Simple CNN with 3 conv layers - sufficient for binary task
2. ReLU activation - prevents vanishing gradient
3. MaxPooling - reduces spatial dimensions and computational cost
4. Dropout - prevents overfitting
5. Binary cross-entropy loss - appropriate for binary classification
"""


class BreastCNNModel(nn.Module):
    def __init__(self):
        super(BreastCNNModel, self).__init__()
        # Simpler CNN for binary classification
        self.conv_layers = nn.Sequential(
            # First conv block - extract basic features
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second conv block - increase feature complexity
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third conv block - final feature extraction
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(512, 1),
            nn.Sigmoid()  # Binary classification output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)  # Add this line to check the shape
        return self.fc_layers(x)


if __name__ == "__main__":
    from main import DataManager  # Import DataManager from main.py
    data_manager = DataManager(data_dir="../Datasets")  # Pass parent directory as data_dir
    breast_data = data_manager.load_breast_data()
