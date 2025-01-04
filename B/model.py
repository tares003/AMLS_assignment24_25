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


class BloodModel(nn.Module):
    def __init__(self):
        super(BloodModel, self).__init__()
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


class BloodTrainer:
    def __init__(self, learning_rate=0.001, batch_size=64, num_epochs=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BloodModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self, train_data, val_data):
        train_loader = self._prepare_dataloader(train_data)
        val_loader = self._prepare_dataloader(val_data)

        best_val_acc = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.squeeze()).sum().item()

            # Validation phase
            val_acc = self._validate(val_loader)

            print(f'Epoch {epoch + 1}/{self.num_epochs}:')
            print(f'Training Loss: {train_loss / len(train_loader):.4f}')
            print(f'Training Accuracy: {100. * correct / total:.2f}%')
            print(f'Validation Accuracy: {val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'B/best_blood_model.pth')

    def _prepare_dataloader(self, data):
        images = torch.FloatTensor(data['images']).permute(0, 3, 1, 2)  # Ensure correct shape
        labels = torch.LongTensor(data['labels'])
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.squeeze()).sum().item()

        return 100. * correct / total

    def predict(self, test_data):
        test_loader = self._prepare_dataloader({'images': test_data, 'labels': np.zeros(len(test_data))})
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

if __name__ == "__main__":
    from main import DataManager  # Import DataManager from main.py
    data_manager = DataManager(data_dir="../Datasets")  # Pass parent directory as data_dir
    blood_data = data_manager.load_blood_data()

    blood_trainer = BloodTrainer(
        learning_rate=0.001,  # Small learning rate for stable training
        batch_size=64,  # Larger batch size for multi-class task
        num_epochs=20  # More epochs for complex task
    )

    blood_trainer.train(
        train_data={'images': blood_data['train_images'], 'labels': blood_data['train_labels']},
        val_data={'images': blood_data['val_images'], 'labels': blood_data['val_labels']}
    )