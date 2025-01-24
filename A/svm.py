import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SVMModel:
    def __init__(self, output_classes=2):
        self.output_classes = output_classes
        self.scaler = StandardScaler()
        self.model = SVC(probability=True)
        self.train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'conf_matrix': []}
        self.val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'conf_matrix': []}

    def train(self, train_data, val_data):
        X_train = train_data['images'].reshape(len(train_data['images']), -1)
        y_train = train_data['labels'].ravel()
        X_train = self.scaler.fit_transform(X_train)

        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear']
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        # Store training metrics
        train_metrics = self.evaluate(train_data)
        for key in self.train_metrics:
            self.train_metrics[key].append(train_metrics[key])

        # Store validation metrics
        val_metrics = self.evaluate(val_data)
        for key in self.val_metrics:
            self.val_metrics[key].append(val_metrics[key])

        # Print training progress
        print("Training Progress:")
        for epoch in range(len(self.train_metrics['accuracy'])):
            print(f"Epoch {epoch + 1}:")
            print(f"  Training Accuracy: {self.train_metrics['accuracy'][epoch]:.4f}")
            print(f"  Validation Accuracy: {self.val_metrics['accuracy'][epoch]:.4f}")
            print(f"  Training Precision: {self.train_metrics['precision'][epoch]:.4f}")
            print(f"  Validation Precision: {self.val_metrics['precision'][epoch]:.4f}")
            print(f"  Training Recall: {self.train_metrics['recall'][epoch]:.4f}")
            print(f"  Validation Recall: {self.val_metrics['recall'][epoch]:.4f}")
            print(f"  Training F1 Score: {self.train_metrics['f1'][epoch]:.4f}")
            print(f"  Validation F1 Score: {self.val_metrics['f1'][epoch]:.4f}")

    def evaluate(self, data):
        X = data['images'].reshape(len(data['images']), -1)
        y = data['labels'].ravel()
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
            'conf_matrix': confusion_matrix(y, y_pred),
            'report': classification_report(y, y_pred)
        }
        return metrics

    def plot_metrics(self):
        epochs = range(1, len(self.train_metrics['accuracy']) + 1)
        # Confusion Matrix
        cm = self.val_metrics['conf_matrix'][-1]
        plt.subplot(2, 3, 5)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        plt.tight_layout()
        plt.show()