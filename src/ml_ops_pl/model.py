"""
MNIST classifier using PyTorch Lightning.

PyTorch Lightning is a framework that wraps PyTorch to simplify training loops.
Instead of writing a custom training loop, we define the model structure and
key methods (training_step, validation_step, etc.), and Lightning handles the rest:
    - Training loop iteration
    - Device management (GPU/CPU/TPU)
    - Gradient accumulation
    - Distributed training
    - Logging and checkpointing
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class MyAwesomeModel(pl.LightningModule):
    """
    MNIST classifier as a LightningModule.

    LightningModule is the core abstraction in PyTorch Lightning. It wraps a PyTorch
    nn.Module and requires implementing a few key methods:
    - __init__: Define layers and loss function
    - forward: Define the forward pass (same as regular PyTorch)
    - training_step: What to do with one batch during training
    - validation_step: What to do with one batch during validation
    - test_step: What to do with one batch during testing
    - configure_optimizers: Set up the optimizer(s)

    The Trainer class (see train.py) automatically calls these methods in the right
    order, handles device placement, and logs metrics.
    """

    def __init__(self, lr: float = 1e-3) -> None:
        """Initialize the model architecture and training components.

        Args:
            lr: Learning rate for the optimizer. Default 1e-3 = 0.001.
        """
        super().__init__()

        # save_hyperparameters() is a Lightning convenience that automatically saves
        # all arguments to self.hparams. This allows us to access them later via
        # self.hparams["lr"] (useful for logging and resuming from checkpoints).
        self.save_hyperparameters()

        # Define CNN architecture: 1 input channel (grayscale) -> 3 conv layers -> FC
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 1 -> 32 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 32 -> 64 channels
        self.conv3 = nn.Conv2d(64, 128, 3, 1)  # 64 -> 128 channels
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(128, 10)  # 128 -> 10 output classes (digits 0-9)

        # Loss function: CrossEntropyLoss combines softmax and NLL loss
        self.loss_fn = nn.CrossEntropyLoss()

        # TorchMetrics provides efficient metric computation across distributed setups.
        # We maintain separate accuracy metrics for train/val/test to track performance.
        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.val_acc = MulticlassAccuracy(num_classes=10)
        self.test_acc = MulticlassAccuracy(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        This method is identical to regular PyTorch modules. Lightning doesn't
        change how the forward pass works—it only manages training/val/test loops.

        Architecture:
        Input (1, 28, 28) -> Conv -> Pool -> Conv -> Pool -> Conv -> Pool -> FC -> Output (10,)

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Logits of shape (batch_size, 10) for each digit class
        """
        # Conv layer 1: apply convolution, ReLU activation, max pooling
        x = torch.relu(self.conv1(x))  # Apply 32 filters of size 3x3
        x = torch.max_pool2d(x, 2, 2)  # Reduce spatial dims by 2x2 pooling

        # Conv layer 2
        x = torch.relu(self.conv2(x))  # Apply 64 filters
        x = torch.max_pool2d(x, 2, 2)

        # Conv layer 3
        x = torch.relu(self.conv3(x))  # Apply 128 filters
        x = torch.max_pool2d(x, 2, 2)

        # Flatten to 1D for fully connected layer
        x = torch.flatten(x, 1)  # Keep batch dim, flatten rest

        # Regularization: randomly zero out neurons during training to prevent overfitting
        x = self.dropout(x)

        # Final classification layer: 128 neurons -> 10 class logits
        return self.fc1(x)

    def training_step(self, batch):
        """Process a single batch during training.

        Lightning calls this automatically for each batch in the training loop.
        We return the loss, and Lightning handles backprop and optimizer step.

        Args:
            batch: Tuple of (images, targets) from the dataloader

        Returns:
            Loss value (scalar tensor) for backpropagation
        """
        img, target = batch  # Unpack batch from dataloader

        # Forward pass: compute predictions
        logits = self(img)  # Shape: (batch_size, 10)

        # Compute loss: measures how wrong our predictions are
        loss = self.loss_fn(logits, target)  # Scalar value

        # Compute accuracy: what fraction of predictions are correct?
        acc = self.train_acc(logits, target)  # Value between 0 and 1

        # Log metrics to tensorboard/CSV/WandB via Lightning's logging system.
        # prog_bar=True shows these in the progress bar during training.
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        # Return loss: Lightning automatically calls .backward() on this
        # and steps the optimizer (see configure_optimizers).
        return loss

    def validation_step(self, batch, batch_idx: int):
        """Process a single batch during validation.

        Lightning calls this automatically for each batch in the validation loop,
        typically at the end of each training epoch. Note: model is in eval mode,
        dropout is disabled, and gradients are not computed.

        Args:
            batch: Tuple of (images, targets) from the validation dataloader
            batch_idx: Index of the batch (not used, but required by Lightning)

        Returns:
            None (metrics are logged, not returned)
        """
        img, target = batch
        logits = self(img)
        loss = self.loss_fn(logits, target)
        acc = self.val_acc(logits, target)

        # Log metrics. Key differences from training_step:
        # - on_epoch=True: aggregate metrics over the whole validation epoch
        # - sync_dist=False: don't synchronize across GPUs (Lightning default)
        # Logging without a return value is fine for val/test; only training_step needs a loss return
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=False)

    def test_step(self, batch, batch_idx: int):
        """Process a single batch during testing.

        Lightning calls this when trainer.test() is invoked. Same setup as
        validation (eval mode, no gradients), but typically on a held-out test set.

        Args:
            batch: Tuple of (images, targets) from the test dataloader
            batch_idx: Index of the batch (not used, but required by Lightning)
        """
        img, target = batch
        logits = self(img)
        loss = self.loss_fn(logits, target)
        acc = self.test_acc(logits, target)

        # Log test metrics separately so they don't mix with validation metrics
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        """Set up the optimizer(s).

        Lightning calls this once before training to get the optimizer.
        By centralizing optimizer setup here, Lightning can better manage:
        - Learning rate scheduling
        - Distributed training with multiple optimizers
        - Mixed precision training

        Returns:
            torch.optim.Optimizer: The optimizer to use (here: Adam)
        """
        # Adam is an adaptive learning rate optimizer that often works well with minimal tuning.
        # We use self.hparams["lr"] to access the learning rate saved in __init__.
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])


if __name__ == "__main__":
    # Quick sanity check: ensure the model can do a forward pass without errors
    model = MyAwesomeModel()
    print(f"Model architecture:\n{model}")
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with a dummy batch of 1 image
    dummy_input = torch.randn(1, 1, 28, 28)  # Shape: (batch=1, channels=1, H=28, W=28)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")  # Should be (1, 10) for 10 classes
