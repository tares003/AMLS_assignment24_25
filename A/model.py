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


class BreastModel(nn.Module):
    def __init__(self):
        super(BreastModel, self).__init__()
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
            nn.Linear(128 * 16 * 16, 512),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(512, 1),
            nn.Sigmoid()  # Binary classification output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)  # Add this line to check the shape
        return self.fc_layers(x)


class BreastTrainer:
    def __init__(self, learning_rate=0.0001, batch_size=10, num_epochs=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BreastModel().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self, train_data, val_data):
        # Prepare data loaders
        train_loader = self._prepare_dataloader(train_data)
        val_loader = self._prepare_dataloader(val_data)

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation phase
            val_loss = self._validate(val_loader)

            print(f'Epoch {epoch + 1}/{self.num_epochs}:')
            print(f'Training Loss: {train_loss / len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), './best_breast_model.pth')

    def _prepare_dataloader(self, data):
        images = torch.FloatTensor(data['images']).unsqueeze(1)  # Add channel dimension
        labels = torch.FloatTensor(data['labels'])
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, labels.float()).item()
        return val_loss / len(val_loader)

    def predict(self, test_data):
        test_loader = self._prepare_dataloader({'images': test_data, 'labels': np.zeros(len(test_data))})
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.extend((outputs > 0.5).cpu().numpy())

        return np.array(predictions)


if __name__ == "__main__":
    from main import DataManager  # Import DataManager from main.py
    data_manager = DataManager(data_dir="../Datasets")  # Pass parent directory as data_dir
    breast_data = data_manager.load_breast_data()

    breast_trainer = BreastTrainer(
        learning_rate=0.001,
        batch_size=10,
        num_epochs=50
    )

    breast_trainer.train(
        train_data={'images': breast_data['train_images'], 'labels': breast_data['train_labels']},
        val_data={'images': breast_data['val_images'], 'labels': breast_data['val_labels']}
    )