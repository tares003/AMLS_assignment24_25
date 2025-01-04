# main.py
"""
ELEC0134 (24/25) - Applied Machine Learning Systems
UCL Department of Electronic and Electrical Engineering

This module implements the data management system for the AMLS assignment, handling both
BreastMNIST and BloodMNIST datasets from the MedMNIST database.

The implementation supports both local data loading and automatic downloading through
the medmnist package, ensuring efficient data handling for machine learning tasks.

Author: [Your Name]
Student Number: [Your Student Number]
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
from medmnist import BloodMNIST, BreastMNIST
from torchvision import transforms

# Configure logging with timestamp and level for debugging purposes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataManager:
    """
    A class to manage the loading and preprocessing of medical image datasets.

    This class handles both BreastMNIST (binary classification) and BloodMNIST
    (multi-class classification) datasets, providing functionality for both local
    file loading and automatic downloading through the medmnist package.

    Attributes:
        data_dir (Path): Directory path for storing dataset files
        breast_path (Path): Path to BreastMNIST dataset file
        blood_path (Path): Path to BloodMNIST dataset file
    """

    def __init__(self, data_dir="Datasets"):
        """
        Initialize the DataManager with the specified data directory.

        Args:
            data_dir (str): Path to the directory where datasets will be stored
                          Default is "Datasets"
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Define paths for both datasets
        self.breast_path = self.data_dir / "breastmnist_128.npz"
        self.blood_path = self.data_dir / "bloodmnist_128.npz"

    def validate_data(self, data: Dict[str, npt.NDArray]) -> bool:
        """Validate loaded data for consistency and correctness."""
        try:
            expected_keys = {'train_images', 'train_labels', 'val_images',
                             'val_labels', 'test_images', 'test_labels'}
            if not all(key in data for key in expected_keys):
                return False

            # Validate shapes and data types
            if not all(data[f'{split}_images'].shape[0] == data[f'{split}_labels'].shape[0]
                       for split in ['train', 'val', 'test']):
                return False

            return True
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False

    def preprocess_images(self, images: npt.NDArray) -> npt.NDArray:
        """Preprocess images by normalizing and standardizing."""
        return images.astype('float32') / 255.0

    def load_breast_data(self) -> Dict[str, npt.NDArray]:
        """
        Load the BreastMNIST dataset either from local storage or download if necessary.

        The dataset contains breast ultrasound images for binary classification:
        - Benign (non-cancerous)
        - Malignant (cancerous)

        Returns:
            dict: Dictionary containing training, validation, and test sets with their labels
                 Format: {
                     'train_images': ndarray, 'train_labels': ndarray,
                     'val_images': ndarray, 'val_labels': ndarray,
                     'test_images': ndarray, 'test_labels': ndarray
                 }

        Raises:
            Exception: If there's an error in loading or downloading the dataset
        """
        try:
            if self.breast_path.exists():
                logger.info("Loading BreastMNIST from local file...")
                data = np.load(self.breast_path)
                data_dict = {
                    'train_images': data['train_images'],
                    'train_labels': data['train_labels'],
                    'val_images': data['val_images'],
                    'val_labels': data['val_labels'],
                    'test_images': data['test_images'],
                    'test_labels': data['test_labels']
                }
            else:
                logger.info("Downloading BreastMNIST dataset...")
                # Download and split the dataset using medmnist
                dataset = BreastMNIST(split='train', download=True)
                val_dataset = BreastMNIST(split='val', download=True)
                test_dataset = BreastMNIST(split='test', download=True)

                # Save to local file for future use
                np.savez(
                    self.breast_path,
                    train_images=dataset.imgs,
                    train_labels=dataset.labels,
                    val_images=val_dataset.imgs,
                    val_labels=val_dataset.labels,
                    test_images=test_dataset.imgs,
                    test_labels=test_dataset.labels
                )

                return self.load_breast_data()

            if not self.validate_data(data_dict):
                raise ValueError("Invalid data format")
            # Total data set
            print("Total images",
                  len(data_dict['train_images']) + len(data_dict['val_images']) + len(data_dict['test_images']))
            print("Train images", len(data_dict['train_images']))
            print("Val images", len(data_dict['val_images']))
            print("Test images", len(data_dict['test_images']))

            # Preprocess images
            for key in ['train_images', 'val_images', 'test_images']:
                data_dict[key] = self.preprocess_images(data_dict[key])

            # Augment data
            for key in ['train', 'val', 'test']:
                augmented_data = self.augment_breast_data(data_dict[f'{key}_images'], data_dict[f'{key}_labels'],
                                                          increase_by=0.5)
                data_dict[f'{key}_images'] = augmented_data['images']
                data_dict[f'{key}_labels'] = augmented_data['labels']

            print("TOTAL IMAGES AFTER AUGMENTATION:",
                  len(data_dict['train_images']) + len(data_dict['val_images']) + len(data_dict['test_images']))
            print("Train images", len(data_dict['train_images']))
            print("Val images", len(data_dict['val_images']))
            print("Test images", len(data_dict['test_images']))

            return data_dict

        except Exception as e:
            logger.error(f"Error loading BreastMNIST dataset: {e}")
            raise

    def augment_breast_data(self, images: np.ndarray, labels: np.ndarray, increase_by: float, seed: int = 42) -> Dict[
        str, np.ndarray]:
        """
        Apply data augmentation to the BreastMNIST dataset to increase the number of samples by a given percentage.

        Args:
            images (np.ndarray): Original images.
            labels (np.ndarray): Corresponding labels for the images.
            increase_by (float): Percentage to increase the dataset by (e.g., 0.5 for 50%).
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Dictionary containing augmented images and labels.
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Define the transformations to apply to the images
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
            transforms.RandomRotation(20),  # Randomly rotate the image by up to 20 degrees
            transforms.ToTensor()  # Convert PIL Image back to tensor
        ])

        # Calculate the number of augmented images to create
        num_original_images = len(images)
        num_augmented_images = int(num_original_images * increase_by)

        augmented_images = []
        augmented_labels = []

        # Create augmented images
        for _ in range(num_augmented_images):
            idx = np.random.randint(0, num_original_images)  # Randomly select an original image
            img, label = images[idx], labels[idx]
            img_tensor = torch.tensor(img).unsqueeze(0)  # Add channel dimension
            augmented_img = transform(img_tensor).squeeze(
                0).numpy()  # Apply transformations and convert back to numpy array
            augmented_images.append(augmented_img)
            augmented_labels.append(label)

        # Convert lists to numpy arrays
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)

        # Concatenate original and augmented data
        return {
            'images': np.concatenate((images, augmented_images), axis=0),
            'labels': np.concatenate((labels, augmented_labels), axis=0)
        }

    def load_blood_data(self) -> Dict[str, npt.NDArray]:
        """
        Load the BloodMNIST dataset either from local storage or download if necessary.

        The dataset contains blood cell images for multi-class classification
        with 8 different types of blood cells.

        Returns:
            dict: Dictionary containing training, validation, and test sets with their labels
                 Format: {
                     'train_images': ndarray, 'train_labels': ndarray,
                     'val_images': ndarray, 'val_labels': ndarray,
                     'test_images': ndarray, 'test_labels': ndarray
                 }

        Raises:
            Exception: If there's an error in loading or downloading the dataset
        """
        try:
            if self.blood_path.exists():
                logger.info("Loading BloodMNIST from local file...")
                data = np.load(self.blood_path)
                data_dict = {
                    'train_images': data['train_images'],
                    'train_labels': data['train_labels'],
                    'val_images': data['val_images'],
                    'val_labels': data['val_labels'],
                    'test_images': data['test_images'],
                    'test_labels': data['test_labels']
                }
            else:
                logger.info("Downloading BloodMNIST dataset...")
                # Download and split the dataset using medmnist
                dataset = BloodMNIST(split='train', download=True)
                val_dataset = BloodMNIST(split='val', download=True)
                test_dataset = BloodMNIST(split='test', download=True)

                # Save to local file for future use
                np.savez(
                    self.blood_path,
                    train_images=dataset.imgs,
                    train_labels=dataset.labels,
                    val_images=val_dataset.imgs,
                    val_labels=val_dataset.labels,
                    test_images=test_dataset.imgs,
                    test_labels=test_dataset.labels
                )

                return self.load_blood_data()

            if not self.validate_data(data_dict):
                raise ValueError("Invalid data format")

            # Preprocess images
            for key in ['train_images', 'val_images', 'test_images']:
                data_dict[key] = self.preprocess_images(data_dict[key])

            return data_dict

        except Exception as e:
            logger.error(f"Error loading BloodMNIST dataset: {e}")
            raise


def main():
    """
    Main execution function for the data management system.

    This function:
    1. Initializes the DataManager
    2. Loads both datasets
    3. Verifies the data shapes
    4. Logs the results

    The function serves as a verification tool to ensure proper data loading
    and processing before model training.
    """
    # Initialize data manager
    data_manager = DataManager()

    # Load datasets
    try:
        breast_data = data_manager.load_breast_data()
        blood_data = data_manager.load_blood_data()

        # Print dataset shapes to verify loading
        logger.info("BreastMNIST dataset shapes:")
        logger.info(f"Training set: {breast_data['train_images'].shape}")
        logger.info(f"Validation set: {breast_data['val_images'].shape}")
        logger.info(f"Test set: {breast_data['test_images'].shape}")

        logger.info("\nBloodMNIST dataset shapes:")
        logger.info(f"Training set: {blood_data['train_images'].shape}")
        logger.info(f"Validation set: {blood_data['val_images'].shape}")
        logger.info(f"Test set: {blood_data['test_images'].shape}")

        # Additional verification steps could be added here


    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


from A.model import BreastTrainer

def train_models():
    """
    Train both BreastMNIST and BloodMNIST models using DataManager for data loading.
    """
    # Initialize DataManager and load datasets
    data_manager = DataManager()
    breast_data = data_manager.load_breast_data()
    blood_data = data_manager.load_blood_data()

    # Initialize trainers
    breast_trainer = BreastTrainer(
        learning_rate=0.001,  # Small learning rate for stable training
        batch_size=32,  # Smaller batch size for better generalization
        num_epochs=10  # Sufficient epochs for convergence
    )


    # Train BreastMNIST model
    logging.info("Training BreastMNIST model...")
    breast_trainer.train(
        train_data={'images': breast_data['train_images'], 'labels': breast_data['train_labels']},
        val_data={'images': breast_data['val_images'], 'labels': breast_data['val_labels']}
    )

if __name__ == "__main__":
    train_models()
    # main()
