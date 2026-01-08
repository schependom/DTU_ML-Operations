"""Evaluation script for a trained MNIST model.

This script demonstrates how to:
1. Load a checkpoint saved during training
2. Use a Trainer to run the test loop
3. Evaluate model performance on the test set

After training, use this script to get final metrics on held-out test data.
Example: python -m ml_ops_pl.evaluate --checkpoint-path models/mnist-epoch=04-val_acc=0.991.ckpt
"""

from __future__ import annotations

import pytorch_lightning as pl
import typer

from ml_ops_pl.data import CorruptMNISTDataModule, DataConfig
from ml_ops_pl.model import MyAwesomeModel


def main(
    checkpoint_path: str,
    data_dir: str = "data/processed",
    batch_size: int = 64,
    num_workers: int = 0,
    val_split: float = 0.1,
) -> None:
    """Evaluate a trained model on the test set.

    Loads a checkpoint from training and runs the test loop to report
    final metrics (test_loss, test_acc).

    Args:
        checkpoint_path: Path to the model checkpoint file (.ckpt)
        data_dir: Path to directory with processed MNIST tensors
        batch_size: Number of images per batch
        num_workers: Number of worker processes for data loading
        val_split: Validation split (used to recreate datamodule, not for testing)
    """
    # --- Create Data Module ---
    # We recreate the data module with the same config used during training.
    config = DataConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
    )
    datamodule = CorruptMNISTDataModule(config)

    # --- Load Trained Model from Checkpoint ---
    # Lightning checkpoints save both the model weights and hyperparameters.
    # load_from_checkpoint() reconstructs the full LightningModule, including
    # the architecture and learning rate used during training.
    model = MyAwesomeModel.load_from_checkpoint(checkpoint_path)

    # --- Create Trainer for Testing ---
    # We create a minimal Trainer just for running the test loop.
    # No need for loggers or callbacks, we just want to run test_step on all test batches.
    trainer = pl.Trainer(accelerator="auto", devices=1)

    # --- Test ---
    # trainer.test() runs the model through the test dataloader and calls
    # model.test_step() for each batch. Metrics are logged and printed.
    # Unlike training, there's no backpropagation, just forward passes in eval mode.
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    typer.run(main)
