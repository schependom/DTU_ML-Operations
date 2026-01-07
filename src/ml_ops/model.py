from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, model_conf: Optional[DictConfig] = None) -> None:
        super().__init__()
        if model_conf is None:
            # Default values for backward compatibility
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.conv3 = nn.Conv2d(64, 128, 3, 1)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128, 10)
        else:
            # Initialize from config
            self.conv1 = nn.Conv2d(
                model_conf.conv1.in_channels,
                model_conf.conv1.out_channels,
                model_conf.conv1.kernel_size,
                model_conf.conv1.stride,
            )
            self.conv2 = nn.Conv2d(
                model_conf.conv2.in_channels,
                model_conf.conv2.out_channels,
                model_conf.conv2.kernel_size,
                model_conf.conv2.stride,
            )
            self.conv3 = nn.Conv2d(
                model_conf.conv3.in_channels,
                model_conf.conv3.out_channels,
                model_conf.conv3.kernel_size,
                model_conf.conv3.stride,
            )
            self.dropout = nn.Dropout(model_conf.dropout_rate)
            self.fc1 = nn.Linear(
                model_conf.fc1.in_features,
                model_conf.fc1.out_features,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


#########
# Commenting best practice (1):
#  add tensor dimensionality to code
#########

#       x = torch.randn(5, 10)                              # N x D
#       y = torch.randn(7, 10)                              # M x D
#       xy = x.unsqueeze(1) - y.unsqueeze(0)                # (N x 1 x D) - (1 x M x D) = (N x M x D)
#       pairwise_euc_dist = xy.abs().pow(2.0).sum(dim=-1)   # N x M

########
# How do these dimensionality manipulations work?
########

#### 1. Unsqueeze
#
# Adds a dimension of size 1 at the specified position.
# Example: If x has shape (5, 10), then x.unsqueeze(1) will have shape (5, 1, 10).

#### 2. Broadcasting
#
# When performing operations on tensors x and y of different shapes,
#  PyTorch looks at the dimensions FROM RIGHT to LEFT (!) and applies the following rules:
#  - If the dimensions are equal, they are compatible.
#  - If one of the dimensions is 1, it is stretched to match the other dimension.
#  - If the dimensions are different and neither is 1, an error is raised.
#
# Example:
#
#   In the expression x.unsqueeze(1) - y.unsqueeze(0),
#   x.unsqueeze(1) has shape (N, 1, D) and y.unsqueeze(0) has shape (1, M, D).
#
#  The dimensions are compared as follows:
#
#   - Dim 2
#     - D vs D
#     - equal, compatible
#     - Resulting dim: D
#
#   - Dim 1
#     - 1 vs M
#     - `x` has size 1. PyTorch copies the row of `x` M times.
#     - Resulting dim: M
#
#   - Dim 0
#     - N vs 1
#     - `y` has size 1. PyTorch copies the column of `y` N times.
#     - Resulting dim: N
#
#  Resulting shape after broadcasting: (N, M, D).
#  The output tensor xy at index [i, j, :] contains the vector result of x[i]−y[j]


#########
# Commenting best practice (2):
#  add docstrings to functions and classes
#########


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
