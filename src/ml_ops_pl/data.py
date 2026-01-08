"""Data loading and preprocessing for Corrupt MNIST with PyTorch Lightning.

LightningDataModule is Lightning's abstraction for data handling. Instead of manually
creating dataloaders in your training code, you define them once in a DataModule.
This makes it easy to:
- Switch datasets without changing training code
- Handle data preprocessing in one place
- Share data definitions across train/val/test
- Support distributed data loading

Key concept: The setup() method is called once at the start of training and creates
the actual train/val/test datasets. The dataloader methods are called each time
Lightning needs to iterate over data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


def _load_split(path: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images and targets for a data split from disk.

    Args:
        path: Directory containing the tensor files
        split: One of "train" or "test"

    Returns:
        Tuple of (images, targets) tensors
    """
    images = torch.load(path / f"{split}_images.pt")
    targets = torch.load(path / f"{split}_target.pt")
    return images, targets


@dataclass
class DataConfig:
    """Configuration for the data module.

    Using a dataclass makes configuration explicit and type-safe. This gets
    passed to CorruptMNISTDataModule to control batch size, data paths, etc.
    """

    data_dir: str = "data/processed"  # Directory with train_images.pt, etc.
    batch_size: int = 64  # Number of samples per batch
    num_workers: int = 0  # Number of processes for data loading (0 = main process)
    val_split: float = 0.1  # Fraction of training data to use for validation


class CorruptMNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Corrupt MNIST tensors.

    A LightningDataModule handles all data-related logic. It works with the
    Trainer to automatically feed data to the model during train/val/test.

    Key Lightning integration points:
    - setup(): Called once before training. Creates self.train_set, val_set, test_set
    - train_dataloader(): Called at each epoch to iterate over training data
    - val_dataloader(): Called at end of each epoch for validation
    - test_dataloader(): Called when trainer.test() is invoked

    See: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        """Initialize the data module.

        Args:
            config: DataConfig object with batch size, data dir, etc.
                   If None, uses defaults from DataConfig().
        """
        super().__init__()

        # If system has more than 0 CPUs and num_workers not set,
        # use up to 4 workers for data loading
        num_workers = config.num_workers if config and config.num_workers > 0 else min(4, os.cpu_count() or 1)

        self.config = config or DataConfig(num_workers=num_workers)
        self.data_dir = Path(self.config.data_dir)

        # These will be populated in setup()
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        """Prepare data for training/validation/testing.

        Lightning calls this once before training. The 'stage' parameter lets you
        differentiate setup for train vs test if needed (e.g., loading only test
        data for evaluation).

        Args:
            stage: One of "fit", "validate", "test", or None (ignored here)
        """
        # Load raw tensors from disk
        train_images, train_targets = _load_split(self.data_dir, "train")
        test_images, test_targets = _load_split(self.data_dir, "test")

        # Data is already normalized and has channel dimension from preprocessing,
        # so we just ensure correct dtypes
        train_targets = train_targets.long()
        test_targets = test_targets.long()

        # Create a TensorDataset: combines images and targets into one dataset
        # TensorDataset returns (image, target) tuples when indexed
        full_train = TensorDataset(train_images, train_targets)

        # Split training data into train and validation sets.
        # This lets us monitor performance on unseen data during training.
        # Example: if val_split=0.1, use 90% for training, 10% for validation
        val_len = int(len(full_train) * self.config.val_split)
        train_len = len(full_train) - val_len

        # random_split() shuffles indices before splitting to avoid sequential bias
        self.train_set, self.val_set = random_split(full_train, [train_len, val_len])

        # Create test dataset (separate from training data)
        self.test_set = TensorDataset(test_images, test_targets)

    def train_dataloader(self) -> DataLoader:
        """Return dataloader for training data.

        Lightning calls this at the start of each epoch. The DataLoader:
        - Groups samples into batches
        - Shuffles them (good for training stability)
        - Prefetches data asynchronously (pin_memory)

        Returns:
            DataLoader that yields (images, targets) batches
        """
        assert self.train_set is not None
        return DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            shuffle=True,  # Important: randomize order to prevent overfitting
            num_workers=self.config.num_workers,
            pin_memory=True,  # Speeds up data transfer to GPU
        )

    def val_dataloader(self) -> DataLoader:
        """Return dataloader for validation data.

        Lightning calls this at the end of each epoch. Unlike training:
        - shuffle=False: evaluate on data in consistent order
        - No gradient computation (Lightning handles this)

        Returns:
            DataLoader that yields (images, targets) batches
        """
        assert self.val_set is not None
        return DataLoader(
            self.val_set,
            batch_size=self.config.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return dataloader for test data.

        Lightning calls this when trainer.test() is invoked. This should be
        your held-out test set used only for final evaluation.

        Returns:
            DataLoader that yields (images, targets) batches
        """
        assert self.test_set is not None
        return DataLoader(
            self.test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
