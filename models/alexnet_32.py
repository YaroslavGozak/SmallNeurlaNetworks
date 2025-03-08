# Import the libraries we need to use in this lab

# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to use arrays to manipulate and store data
import numpy as np

from utils import _log_api_usage_once
from models.small_cnn_generic_2_layers import Small_CNN_Generic_2_layers


class AlexNet32(Small_CNN_Generic_2_layers):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__(11, 5, [3,64,192,384,256,256], 32, 1, 1)
        _log_api_usage_once(self)
        self.features = nn.Sequential(
            # 32x32x3
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5),
            # 32x32x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 16x16x64

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # 16x16x192
            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 8x8x192

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # 8x8x384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # 8x8x256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 8x8x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 4x4x256
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # 1x1x4096
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            # 4096
            nn.Linear(256 * 4 * 4, 4096),
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
        