from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class NoisyCIFAR10(Dataset):
    def __init__(self,
                 root: str,
                 train: bool, 
                 noise_std: float,
                 augment: bool) :
        if augment:
            tf = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            tf = transforms.Compose([transforms.ToTensor()])

        self.base = datasets.CIFAR10(root=root, 
                                     train=train, 
                                     transform=tf)
        self.noise_std = noise_std / 255.0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        clean, label = self.base[idx]
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean, label


def build_denoising_dataloaders(
    root: str,
    noise_std: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) :
    train_set = NoisyCIFAR10(root=root, 
                             train=True, 
                             noise_std=noise_std, 
                             augment=True)
    test_set = NoisyCIFAR10(root=root, 
                            train=False, 
                            noise_std=noise_std, 
                            augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
