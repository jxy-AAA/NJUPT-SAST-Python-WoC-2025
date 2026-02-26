import torch
from torch import nn

from .common import ClassifierHead, SharedEncoder


class Task2ClassifierNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_blocks: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = SharedEncoder(in_channels=in_channels, 
                                     base_channels=base_channels, 
                                     num_blocks=num_blocks)
        self.classifier_head = ClassifierHead(channels=base_channels, 
                                              num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier_head(features)
