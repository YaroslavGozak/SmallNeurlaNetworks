# Import the libraries we need to use in this lab

# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to use arrays to manipulate and store data
import numpy as np

from utils import _log_api_usage_once


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = nn.Sequential(
            # 224x224x3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # 55x55x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27x64

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # 27x27x192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13x13x192

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # 13x13x384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # 13x13x256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 13x13x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 6x6x256
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 1x1x9216
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            # 9216
            nn.Linear(256 * 6 * 6, 4096),
            # 4096
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x