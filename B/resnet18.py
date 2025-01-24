import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, output_classes=8, dropout_rate=0.5):
        super(ResNet18, self).__init__()
        # Use the new weights parameter instead of pretrained
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.resnet.fc = nn.Identity()

        self.custom_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_classes)
        )

        self.final_activation = (nn.Softmax(dim=1) if output_classes > 1 else nn.Sigmoid())

    def forward(self, x):
        x = self.resnet(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1) # Reshape for AdaptiveAvgPool2d
        x = self.custom_head(x)
        if self.training:
            return x
        return self.final_activation(x)
