# Import the libraries we need to use in this lab

# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to use arrays to manipulate and store data
import numpy as np

from utils import _log_api_usage_once


class AlexNet32(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = nn.Sequential(
            # 32x32x3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # 7x7x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 3x3x64

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # 3x3x192
            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 1x1x192

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # 1x1x384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # 1x1x256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 1x1x256
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # # 6x6x256
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 1x1x9216
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            # 9216
            # nn.Linear(256 * 6 * 6, 4096),
            # 256
            nn.Linear(256, 4096),
            # 4096
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        # 1x1x256 -> 256
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        