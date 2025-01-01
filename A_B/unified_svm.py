import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class UnifiedSVMModel:
    def __init__(self, task_type='binary'):
        """
        Initialize SVM model for either binary or multiclass classification
        Args:
            task_type: 'binary' for Task A or 'multiclass' for Task B
        """
        self.task_type = task_type
        self.num_classes = 2 if task_type == 'binary' else 8
        self.scaler = StandardScaler()
        self.model = SVC(probability=True)

    def train(self, train_data):
        """Train SVM model with task-specific configurations"""
        X_train = train_data['images'].reshape(len(train_data['images']), -1)
        y_train = train_data['labels'].ravel()
        X_train = self.scaler.fit_transform(X_train)

        # Adjust parameters based on task
        if self.task_type == 'binary':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None]
            }
        else:  # multiclass
            param_grid = {
                'C': [1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'decision_function_shape': ['ovo', 'ovr']
            }

        # Grid search with stratified k-fold
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='accuracy' if self.task_type == 'binary' else 'f1_macro',
            n_jobs=-1  # Use all available cores
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_

    def predict(self, data):
        """Make predictions with probability scores"""
        X = data['images'].reshape(len(data['images']), -1)
        X = self.scaler.transform(X)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities

    def evaluate(self, data, phase='validation'):
        """Comprehensive evaluation based on task type"""
        X = data['images'].reshape(len(data['images']), -1)
        y_true = data['labels'].ravel()
        X = self.scaler.transform(X)

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        metrics = {}

        # Common metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        if self.task_type == 'binary':
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        else:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        metrics['classification_report'] = classification_report(y_true, y_pred)

        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        plt.imshow(metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {phase}')
        plt.colorbar()
        plt.savefig(f'confusion_matrix_{self.task_type}_{phase}.png')
        plt.close()

        return metrics

    def plot_decision_boundary(self, data, filename=None):
        """Plot decision boundary with appropriate color scheme per task"""
        if self.task_type != 'binary':
            print("Decision boundary plotting is only supported for binary classification")
            return

        X = data['images'].reshape(len(data['images']), -1)
        y = data['labels'].ravel()
        X = self.scaler.transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        h = .02  # step size
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = self.model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.RdYlBu, edcolors='black')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f'SVM Decision Boundary - {self.task_type}')

        if filename:
            plt.savefig(filename)
        plt.close()
