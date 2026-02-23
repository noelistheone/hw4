# =========================
# problem2.py
# =========================
"""
HW4 - Problem 2: CNN on CIFAR-10 (PyTorch)

You will implement a compact CNN, a deterministic corruption function,
and a comparison routine that trains MLP vs CNN under the same budget.

Autograder imports these functions/classes directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# You may import from problem1 for convenience
from problem1 import set_seed, train_one_epoch, evaluate, MLP


class SmallCNN(nn.Module):
    """
    CNN classifier for CIFAR-10.

    Requirements:
    - At least 3 conv layers
    - BatchNorm in at least 2 places
    - At least one pooling op
    - Global pooling / adaptive pooling before classifier
    - Output logits (B, 10)
    """

    def __init__(self, width: int = 32, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(width*2)
        self.conv3 = nn.Conv2d(width*2, width*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(width*4)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(width*4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def apply_corruption(
    x: torch.Tensor,
    kind: str,
    severity: float,
    seed: int,
) -> torch.Tensor:
    """
    Apply a deterministic corruption to images in [0,1].

    Parameters
    ----------
    x : torch.Tensor, shape (B,3,32,32), values in [0,1]
    kind : one of {"gaussian_noise", "channel_drop", "cutout"}
    severity : controls strength/area (interpretation up to you, but must be consistent)
    seed : int for determinism

    Returns
    -------
    x_corr : torch.Tensor, shape (B,3,32,32), values clipped to [0,1]
    """
    # TODO
    set_seed(seed)
    if kind == "gaussian_noise":
        noise = torch.randn_like(x) * severity
        x_corr = x + noise
    elif kind == "channel_drop":
        drop_mask = torch.rand(x.size(0), 3, 1, 1, device=x.device) < severity
        x_corr = x * (~drop_mask)
    elif kind == "cutout":
        cutout_size = int(severity * 32)
        x_corr = x.clone()
        for i in range(x.size(0)):
            x_start = torch.randint(0, 32 - cutout_size, (1,))
            y_start = torch.randint(0, 32 - cutout_size, (1,))
            x_corr[i, :, y_start:y_start+cutout_size, x_start:x_start+cutout_size] = 0
    else:
        raise ValueError(f"Unknown corruption kind: {kind}")
    
    return torch.clamp(x_corr, 0, 1)


def compare_mlp_cnn(
    mlp: nn.Module,
    cnn: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    seed: int,
) -> Dict[str, float]:
    """
    Train both models for the same number of epochs and return:
      {"mlp_test_acc": ..., "cnn_test_acc": ..., "delta": ...}
    where delta = cnn_test_acc - mlp_test_acc.

    You must use the same seed for determinism.
    """
    # TODO
    set_seed(seed)
    opt_mlp = torch.optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)
    opt_cnn = torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)

    for _ in range(epochs):
        train_one_epoch(mlp, train_loader, opt_mlp, device)
        train_one_epoch(cnn, train_loader, opt_cnn, device)

    mlp_test_acc = evaluate(mlp, test_loader, device)["acc"]
    cnn_test_acc = evaluate(cnn, test_loader, device)["acc"]
    return {
        "mlp_test_acc": mlp_test_acc,
        "cnn_test_acc": cnn_test_acc,
        "delta": cnn_test_acc - mlp_test_acc,
    }


if __name__ == "__main__":
    # Quick sanity run (not graded)
    from problem1 import get_cifar10_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        set_seed(0)
        train_loader, test_loader = get_cifar10_loaders(batch_size=64, seed=0, limit_train=512, limit_test=256)
        mlp = MLP(hidden_sizes=(512, 256), dropout_p=0.2).to(device)
        cnn = SmallCNN(width=32, dropout_p=0.1).to(device)
        out = compare_mlp_cnn(mlp, cnn, train_loader, test_loader, device, epochs=1, seed=0)
        print(out)
    except NotImplementedError:
        print("Implement TODOs first.")
