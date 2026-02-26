from typing import Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) :
        return self.relu(x + self.block(x))


class SharedEncoder(nn.Module):
    def __init__(self, in_channels = 3,
                 base_channels = 64, 
                 num_blocks = 6) :
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 
                      base_channels, 
                      kernel_size=3, 
                      padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(num_blocks):
            block = ResidualBlock(base_channels)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x) :
        x = self.stem(x)
        return self.blocks(x)


class DenoiseHead(nn.Module):
    def __init__(self, channels = 64) :
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, 
                      out_channels=3, 
                      kernel_size=3, padding=1),
        )

    def forward(self, features, noisy_input: torch.Tensor) -> torch.Tensor:
        residual = self.net(features)
        denoised = torch.clamp(noisy_input - residual, 0.0, 1.0)
        return denoised


class ClassifierHead(nn.Module):
    def __init__(self, channels = 64, num_classes = 10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features)
        pooled = pooled.flatten(1)
        return self.out(pooled)


class ClassGuidedGate(nn.Module):
    def __init__(self, channels= 64, num_classes = 10) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_classes, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, features, logits: torch.Tensor) -> torch.Tensor:
        gate = self.proj(logits).unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gate)


def classifier_logits_from_features(classifier_head,
                                    features) -> torch.Tensor:
    return classifier_head(features)


def denoise_from_features(denoise_head, 
                          features, 
                          noisy_input) -> torch.Tensor:
    return denoise_head(features, noisy_input)


def make_heads(base_channels: int, num_classes: int):
    denoise_head = DenoiseHead(base_channels)
    classifier_head = ClassifierHead(base_channels, num_classes)
    return denoise_head, classifier_head