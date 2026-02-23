# =========================
# problem1.py
# =========================
"""
HW4 - Problem 1: MLP on CIFAR-10 (PyTorch)

Allowed: numpy/pandas/torch/torchvision.
Do NOT use pretrained models or external training frameworks.

Autograder imports these functions/classes directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


def set_seed(seed: int) -> None:
    """
    Set seeds for random, numpy, and torch to make results deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar10_loaders(
    batch_size: int,
    seed: int,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for CIFAR-10.

    - Images must be float in [0,1].
    - If limit_train is not None, use only the first limit_train training examples.
    - If limit_test is not None, use only the first limit_test test examples.
    - Do not shuffle test loader.
    - Use seed to make selection/shuffle deterministic.
    """
    transform = T.Compose([
        T.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    if limit_train is not None:
        indices = list(range(limit_train))
        train_dataset = Subset(train_dataset, indices=indices)

    if limit_test is not None:
        indices = list(range(limit_test))
        test_dataset = Subset(test_dataset, indices=indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class MLP(nn.Module):
    """
    MLP classifier for CIFAR-10.

    Requirements:
    - Flatten input to (B, 3072)
    - At least two hidden layers
    - ReLU activations
    - Dropout in at least one hidden layer (p provided by constructor)
    - Output logits of shape (B, 10)
    """

    def __init__(self, hidden_sizes=(512, 256), dropout_p: float = 0.2):
        super().__init__()
        # TODO
        layers = []
        input_size = 3072  # CIFAR-10 images are 32x32x3 = 3072 pixels
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 10))  # Output layer for 10 classes
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        x = x.view(x.size(0), -1)  # Flatten input to (B, 3072)
        return self.model(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns dict:
      - "loss": float
      - "acc": float
    """
    # TODO
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return {
        "loss": total_loss / len(loader),
        "acc": correct / total,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model.

    Returns dict:
      - "loss": float
      - "acc": float
    """
    # TODO
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return {
        "loss": total_loss / len(loader),
        "acc": correct / total,
    }


if __name__ == "__main__":
    # Quick sanity run (not graded)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        set_seed(0)
        train_loader, test_loader = get_cifar10_loaders(batch_size=64, seed=0, limit_train=512, limit_test=256)
        model = MLP(hidden_sizes=(512, 256), dropout_p=0.2).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        tr = train_one_epoch(model, train_loader, opt, device)
        te = evaluate(model, test_loader, device)
        print("Train:", tr, "Test:", te)
    except NotImplementedError:
        print("Implement TODOs first.")
