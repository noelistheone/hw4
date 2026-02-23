# =========================
# problem3.py
# =========================
"""
HW4 - Problem 3: RNN on CIFAR-10 images-as-sequences (columns)

Treat each image (B,3,32,32) as a sequence of length 32, where each timestep is a column:
input dim per timestep = 3*32 = 96.

Autograder imports these functions/classes directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


def image_to_column_sequence(x: torch.Tensor) -> torch.Tensor:
    """
    Convert images to column sequences.

    Input
    -----
    x : torch.Tensor, shape (B,3,32,32), values in [0,1]

    Output
    ------
    seq : torch.Tensor, shape (B,32,96)
      timestep t corresponds to column t of the image, flattened across (C,H).
    """
    # TODO
    B, C, H, W = x.shape
    seq = x.permute(0, 3, 1, 2).reshape(B, W, C*H)
    return seq


class ColumnRNN(nn.Module):
    """
    RNN-based classifier for CIFAR-10 using images-as-sequences.

    Requirements:
    - Use nn.RNN or nn.GRU or nn.LSTM
    - Input: (B,32,96)
    - Output logits: (B,10)
    - Use final hidden state (or pooled hidden states) to predict.
    """

    def __init__(
        self,
        rnn_type: str = "gru",   # one of {"rnn","gru","lstm"}
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        # TODO
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=96, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_p)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=96, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_p)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=96, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_p)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x can be either:
          - images: (B,3,32,32), in which case you should call image_to_column_sequence
          - sequences: (B,32,96)
        Return logits: (B,10)
        """
        # TODO
        if x.dim() == 4:
            x = image_to_column_sequence(x)
        output, hidden = self.rnn(x)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        logits = self.fc(hidden[-1])
        return logits


if __name__ == "__main__":
    # Quick sanity run (not graded)
    try:
        B = 4
        x = torch.rand(B, 3, 32, 32)
        seq = image_to_column_sequence(x)
        print("seq shape:", seq.shape)  # (B,32,96)

        model = ColumnRNN(rnn_type="gru", hidden_size=64)
        logits = model(x)
        print("logits shape:", logits.shape)  # (B,10)
    except NotImplementedError:
        print("Implement TODOs first.")
