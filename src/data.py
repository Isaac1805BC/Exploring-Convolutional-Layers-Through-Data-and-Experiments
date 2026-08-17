"""Data loading utilities for the Fashion-MNIST experiments."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Computed once on the raw training split (see notebook EDA section).
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


def get_transform(normalize: bool = True) -> transforms.Compose:
    ops = [transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)))
    return transforms.Compose(ops)


def load_raw_datasets():
    """Return the train/test splits with only ToTensor applied (values in [0, 1])."""
    train_raw = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor()
    )
    test_raw = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    return train_raw, test_raw


def get_dataloaders(
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 42,
    normalize: bool = True,
    num_workers: int = 0,
):
    """Build train/val/test DataLoaders with a fixed train/val split."""
    transform = get_transform(normalize=normalize)

    full_train = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
