from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_classification_dataloaders(
    root: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,) :
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(root=root, 
                                 train=True,
                                 transform=train_tf, 
                                 download=True)
    test_set = datasets.CIFAR10(root=root, 
                                train=False, 
                                transform=test_tf,
                                download=True)

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
