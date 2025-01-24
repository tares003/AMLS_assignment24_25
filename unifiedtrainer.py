import logging
import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedTrainer:
    def __init__(self, model, learning_rate=0.001, batch_size=32, num_epochs=20, task_type='binary'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.task_type = task_type
        self.criterion = nn.BCELoss() if task_type == 'binary' else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.train_times = []
        self.inference_times = []

    def _prepare_dataloader(self, data, is_multiclass=False):
        images = torch.FloatTensor(data['images']).unsqueeze(1) if not is_multiclass else torch.FloatTensor(
            data['images']).permute(0, 3, 1, 2)
        labels = torch.FloatTensor(data['labels']) if self.task_type == 'binary' else torch.LongTensor(
            data['labels']).squeeze()
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, train_data, val_data):
        train_loader = self._prepare_dataloader(train_data, is_multiclass=(self.task_type == 'multiclass'))
        val_loader = self._prepare_dataloader(val_data, is_multiclass=(self.task_type == 'multiclass'))

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float() if self.task_type == 'binary' else labels)
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            if self.task_type == 'multiclass':
                _, predicted = outputs.max(1)
            else:
                predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            train_time = time.time() - start_time
            self.train_times.append(train_time)

            train_accuracy = 100. * correct_train / total_train
            val_loss, val_accuracy = self._validate(val_loader)
            self.train_losses.append(train_loss / len(train_loader))
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            logger.info(f'Epoch {epoch + 1}/{self.num_epochs}:')
            logger.info(f'Training Loss: {train_loss / len(train_loader):.4f}')
            logger.info(f'Training Accuracy: {train_accuracy:.2f}%')
            logger.info(f'Validation Loss: {val_loss:.4f}')
            logger.info(f'Validation Accuracy: {val_accuracy:.2f}%')
            logger.info(f'Training Time: {train_time:.2f}s')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = f'./best_{self.model.__class__.__name__}.pth'
                torch.save(self.model.state_dict(), model_path)
                logger.info(f'Saved best model: {model_path}')

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs, labels.float() if self.task_type == 'binary' else labels).item()
                if self.task_type == 'multiclass':
                    _, predicted = outputs.max(1)
                else:
                    predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_accuracy = 100. * correct_val / total_val
        return val_loss / len(val_loader), val_accuracy

    def plot_metrics(self, val_data):
        epochs = range(1, self.num_epochs + 1)

        plt.figure(figsize=(20, 12))
        plt.suptitle(f'Model - {self.model.__class__.__name__}', fontsize=16)
        plt.subplot(2, 3, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')

        plt.subplot(2, 3, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curve')

        plt.subplot(2, 3, 3)
        plt.plot(epochs, self.learning_rates, label='Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.title('Learning Rate Changes')

        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.train_times, label='Training Time')
        plt.xlabel('Epochs')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.title('Training Time')

        # Confusion Matrix
        val_loader = self._prepare_dataloader(val_data, is_multiclass=(self.task_type == 'multiclass'))
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                if self.task_type == 'multiclass':
                    _, predicted = outputs.max(1)
                else:
                    predicted = (outputs > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.subplot(2, 3, 5)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Precision-Recall Curve and ROC Curve for binary classification
        if self.task_type == 'binary':
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            fpr, tpr, _ = roc_curve(all_labels, all_preds)
            roc_auc = auc(fpr, tpr)

            plt.subplot(2, 3, 6)
            plt.plot(recall, precision, label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.title('Precision-Recall Curve')

            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.title(f'ROC Curve - {self.model.__class__.__name__}')

        plt.tight_layout()
        plt.savefig(f'{self.model.__class__.__name__}_metrics.png')
        plt.show()
