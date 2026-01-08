"""Training script for MNIST with PyTorch Lightning.

This script demonstrates the minimal code needed to train a model with Lightning.
Instead of writing a custom training loop (with gradient accumulation, device
management, logging, etc.), we just:
1. Create a DataModule
2. Create a LightningModule
3. Create a Trainer
4. Call trainer.fit()

Lightning handles all the boilerplate!

Key Lightning concepts used here:
- Trainer: Orchestrates the training loop, validation, logging, checkpointing
- Callbacks: Plugins that hook into the training lifecycle (e.g., ModelCheckpoint)
- Loggers: Save metrics to disk/TensorBoard/WandB (here: CSVLogger)
- accelerator="auto": Automatically detect GPU/TPU/CPU
"""

from __future__ import annotations

import pytorch_lightning as pl
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from ml_ops_pl.data import CorruptMNISTDataModule, DataConfig
from ml_ops_pl.model import MyAwesomeModel


def main(
    data_dir: str = "data/processed",
    batch_size: int = 64,
    num_workers: int = 0,
    val_split: float = 0.1,
    max_epochs: int = 5,
    lr: float = 1e-3,
    checkpoint_dir: str = "./models",
) -> None:
    """Train an MNIST classifier with PyTorch Lightning.

    This script trains a convolutional neural network on Corrupt MNIST using
    PyTorch Lightning, which handles all the training loop boilerplate.

    Args:
        data_dir: Path to directory with processed MNIST tensors
        batch_size: Number of images per batch
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation (0.1 = 10%)
        max_epochs: Number of training epochs
        lr: Learning rate for the optimizer
        checkpoint_dir: Directory to save model checkpoints
    """
    # --- Create Data Module ---
    # The DataModule encapsulates all data loading logic. Lightning will call
    # its methods (setup, train_dataloader, val_dataloader, test_dataloader)
    # automatically at the right times.
    config = DataConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
    )
    datamodule = CorruptMNISTDataModule(config)

    # --- Create Model ---
    # The LightningModule defines the model architecture and training/val/test logic.
    # We pass lr to the constructor so it's saved as a hyperparameter.
    model = MyAwesomeModel(lr=lr)

    # --- Create Callbacks ---
    # Callbacks are plugins that execute code at certain points in the training loop.
    # ModelCheckpoint automatically saves the best model based on a monitored metric.
    callbacks = [
        ModelCheckpoint(
            # Default: after each validation epoch
            # more often: every_n_epochs=1, every_n_train_steps=100, etc.
            # less often: every_n_epochs=5, every_n_train_steps=500, etc.
            dirpath=checkpoint_dir,
            filename="mnist-{epoch:02d}-{val_acc:.3f}",  # Save with epoch and val_acc in filename
            monitor="val_acc",  # Monitor validation accuracy
            mode="max",  # Higher is better (for accuracy)
            save_top_k=1,  # Keep only the best checkpoint
        ),
        EarlyStopping(
            monitor="val_acc",
            patience=3,  # Stop training if no improvement in 3 epochs
            mode="max",  # Higher is better
        ),
    ]

    # --- Create Trainer ---
    # The Trainer is Lightning's main orchestrator. It handles:
    # - Training loop iteration (epochs, batches)
    # - Validation at the end of each epoch
    # - Logging metrics
    # - Device management (GPU/CPU/TPU/MPS)
    # - Checkpointing and resuming
    # - Mixed precision training
    # - Distributed training (with minimal code changes)
    trainer = pl.Trainer(
        accelerator="auto",  # Auto-detect GPU/TPU/CPU
        devices=1,  # Use 1 device (GPU or CPU core)
        max_epochs=max_epochs,  # Stop after this many epochs
        logger=CSVLogger(save_dir="outputs", name="mnist"),  # Log metrics to CSV files
        callbacks=callbacks,  # Attach our checkpoint callback
        log_every_n_steps=10,  # Log metrics every 10 batches
    )

    # --- Train ---
    # This is all you need! The trainer handles the entire training loop.
    # It calls model.training_step() for each batch, model.validation_step()
    # at the end of each epoch, and logs everything automatically.
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    # After training, evaluate on the test set using the best checkpoint.
    # ckpt_path="best" loads the checkpoint that achieved the best validation accuracy.
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    typer.run(main)
