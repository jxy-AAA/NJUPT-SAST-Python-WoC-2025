from typing import Dict, Tuple

import torch
from torch import nn

from .common import ClassGuidedGate, ClassifierHead, DenoiseHead, SharedEncoder


class TaskCommAdd(nn.Module):
    """Bidirectional task communication via 1x1 conv projection + residual add."""

    def __init__(self, denoise_channels: int, classify_channels: int) -> None:
        super().__init__()
        self.cls_to_denoise = nn.Conv2d(classify_channels, 
                                        denoise_channels, 
                                        kernel_size=1, bias=False)
        self.denoise_to_cls = nn.Conv2d(denoise_channels, 
                                        classify_channels, 
                                        kernel_size=1, bias=False)

    def forward(self, denoise_features, classify_features) :
        fused_denoise = denoise_features + self.cls_to_denoise(classify_features)
        fused_classify = classify_features + self.denoise_to_cls(denoise_features)
        return fused_denoise, fused_classify


class MultiTaskNet(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, 
                 num_blocks = 4, num_classes= 10) -> None:
        super().__init__()
        self.encoder = SharedEncoder(in_channels=in_channels, 
                                     base_channels=base_channels, 
                                     num_blocks=num_blocks)
        self.comm = TaskCommAdd(denoise_channels=base_channels, 
                                classify_channels=base_channels)
        self.denoise_head = DenoiseHead(channels=base_channels)
        self.classifier_head = ClassifierHead(channels=base_channels, 
                                              num_classes=num_classes)
        self.gate = ClassGuidedGate(channels=base_channels, 
                                    num_classes=num_classes)

    def forward(self, noisy: torch.Tensor):
        shared_features = self.encoder(noisy)
        denoise_features, classify_features = self.comm(shared_features, 
                                                        shared_features)
        logits = self.classifier_head(classify_features)
        guided_features = self.gate(denoise_features, logits)
        denoised = self.denoise_head(guided_features, noisy)
        return {
            "denoised": denoised,
            "logits": logits,
        }

    def predict_clean_logits(self, clean) -> torch.Tensor:
        clean_features = self.encoder(clean)
        _, classify_features = self.comm(clean_features, clean_features)
        return self.classifier_head(classify_features)

    def freeze_shared_front_half(self) -> None:
        for parameter in self.encoder.stem.parameters():
            parameter.requires_grad = False

        blocks = list(self.encoder.blocks.children())
        num_frozen_blocks = len(blocks) // 2
        for idx, block in enumerate(blocks):
            trainable = idx >= num_frozen_blocks
            for parameter in block.parameters():
                parameter.requires_grad = trainable


def load_single_task_weights(
    model: MultiTaskNet,
    task1_state_dict: Dict[str, torch.Tensor],
    task2_state_dict: Dict[str, torch.Tensor],
) -> None:
    model_state = model.state_dict()

    def _copy_if_match(src_state: Dict[str, torch.Tensor]) -> int:
        copied = 0
        for k, v in src_state.items():
            if k not in model_state:
                continue
            if model_state[k].shape != v.shape:
                continue
            model_state[k] = v
            copied += 1
        return copied

    n1 = _copy_if_match(task1_state_dict)
    n2 = _copy_if_match(task2_state_dict)

    model.load_state_dict(model_state)
    print(f"Loaded from task1: {n1}, task2: {n2}")