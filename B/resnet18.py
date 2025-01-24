import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, output_classes=8, dropout_rate=0.5):
        super(ResNet18, self).__init__()
        # Load pretrained ResNet18 with ImageNet weights
        self.resnet = models.resnet18(pretrained=True)

        # Store original conv1 weights before modification
        original_layer = self.resnet.conv1.weight.data
        # Replace first conv layer to accept 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Convert RGB weights to grayscale by averaging channels
        self.resnet.conv1.weight.data = original_layer.mean(dim=1, keepdim=True)

        # Remove original classifier
        self.resnet.fc = nn.Identity()

        # Custom classification head
        self.custom_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling to reduce spatial dimensions
            nn.Flatten(),  # Flatten for dense layers
            nn.BatchNorm1d(512),  # Normalize features, stabilize training
            nn.Linear(512, 512),  # Dense layer maintaining feature dimension
            nn.ReLU(),  # Non-linear activation
            nn.Dropout(dropout_rate),  # Regularization
            nn.BatchNorm1d(512),  # Additional normalization before final layer
            nn.Linear(512, output_classes)  # Output layer
        )

        # Task-specific activation function
        self.final_activation = (nn.Softmax(dim=1) if output_classes > 1 else nn.Sigmoid())

    def forward(self, x):
        x = self.resnet(x)  # Extract features using ResNet
        x = self.custom_head(x)  # Apply classification head
        if self.training:
            return x  # Return logits for training (better with loss functions)
        return self.final_activation(x)  # Apply activation only during inference
