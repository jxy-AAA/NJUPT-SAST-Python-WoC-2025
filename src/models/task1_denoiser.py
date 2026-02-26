import torch
from torch import nn

from .common import DenoiseHead, SharedEncoder


class Task1DenoiseNet(nn.Module):
    def __init__(self, in_channels = 3, 
                 base_channels = 64, 
                 num_blocks = 4) -> None:
        super().__init__()
        self.encoder = SharedEncoder(in_channels=in_channels, 
                                     base_channels=base_channels, 
                                     num_blocks=num_blocks)
        self.denoise_head = DenoiseHead(channels=base_channels)

    def forward(self, noisy: torch.Tensor) :
        features = self.encoder(noisy)
        return self.denoise_head(features, noisy)
