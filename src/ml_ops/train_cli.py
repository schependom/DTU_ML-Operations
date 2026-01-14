# Training script with CLI instead of Hydra.

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import hydra
import matplotlib.pyplot as plt
import torch
import typer
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb
from ml_ops.data import corrupt_mnist
from ml_ops.device import DEVICE
from ml_ops.model import MyAwesomeModel

# Load environment variables immediately
load_dotenv()

# --- Helper Functions ---


def setup_wandb(cfg: DictConfig) -> bool:
    """Initializes Weights & Biases."""
    if not cfg.wandb.enabled:
        logger.info("WandB disabled via config.")
        return False

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY not found. Skipping WandB.")
        return False

    try:
        wandb.login(key=api_key)

        init_args = {
            "config": cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        }

        # Only set project and entity if not running a sweep (WandB handles this automatically in sweeps)
        if not os.getenv("WANDB_SWEEP_ID"):
            init_args["entity"] = os.getenv("WANDB_ENTITY")
            init_args["project"] = os.getenv("WANDB_PROJECT")

        wandb.init(**init_args)
        logger.success("Initialized Weights & Biases.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return False


def log_sample_images(dataloader: DataLoader) -> None:
    """Logs a sample batch of images to WandB, correcting for normalization."""
    images, _ = next(iter(dataloader))

    # Select first 8 images
    sample_imgs = images[:8]

    # --- VISUALIZATION: Rescale to [0, 1] ---
    # The data is normalized (mean=0, std=1) in data.py, which makes pixels < 0.
    # We rescale using min/max of the batch to ensure values are in [0, 1] for logging.
    sample_imgs = (sample_imgs - sample_imgs.min()) / (sample_imgs.max() - sample_imgs.min())

    # Create a nice grid instead of 8 separate log entries
    grid = make_grid(sample_imgs, nrow=4)

    # Permute from (C, H, W) -> (H, W, C) for WandB/Matplotlib
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    wandb.log({"sample_inputs": wandb.Image(grid_np, caption="Input Data Sample")})
    logger.info("Logged sample images to WandB.")


def save_training_plot(statistics: Dict[str, List[float]], save_path: str) -> None:
    """Saves the loss/accuracy curves to disk."""
    try:
        plt.switch_backend("agg")
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(statistics["train_loss"], label="Train Loss")
        axs[0].set_title("Training Loss")
        axs[0].set_xlabel("Steps")
        axs[0].legend()

        axs[1].plot(statistics["train_accuracy"], label="Train Acc", color="orange")
        axs[1].set_title("Training Accuracy")
        axs[1].set_xlabel("Steps")
        axs[1].legend()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Training plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")


def log_roc_curves(targets: torch.Tensor, preds: torch.Tensor) -> None:
    """Logs ROC curves for each class."""
    try:
        plt.switch_backend("agg")
        fig, ax = plt.subplots(figsize=(10, 8))

        for class_id in range(10):
            # Create binary targets for One-vs-Rest
            one_hot = (targets == class_id).float()
            # Handle cases where a class might not be present in the batch
            if one_hot.sum() > 0:
                RocCurveDisplay.from_predictions(
                    one_hot.cpu().numpy(),
                    preds[:, class_id].cpu().numpy(),
                    name=f"Class {class_id}",
                    ax=ax,
                    plot_chance_level=(class_id == 0),  # Only plot chance once
                )

        wandb.log({"roc_curves": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to log ROC curves: {e}")


# --- Core Logic ---


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    epoch: int,
    cfg: DictConfig,
    use_wandb: bool,
) -> Tuple[List[float], List[float], torch.Tensor, torch.Tensor]:
    """Runs training for a single epoch."""
    model.train()
    batch_losses = []
    batch_accs = []
    all_preds = []
    all_targets = []

    for i, (img, target) in enumerate(dataloader):
        img, target = img.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(img)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()

        # Metrics
        loss_val = loss.item()
        acc_val = (y_pred.argmax(dim=1) == target).float().mean().item()

        batch_losses.append(loss_val)
        batch_accs.append(acc_val)

        # Store for epoch-level metrics (move to CPU to save GPU RAM)
        all_preds.append(y_pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        if use_wandb:
            wandb.log({"train_loss": loss_val, "train_accuracy": acc_val, "epoch": epoch})

            # Log gradients rarely (e.g., once per epoch or very large interval)
            if i == 0:
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.detach().cpu().numpy())})

        if i % cfg.logging.log_interval == 0:
            logger.info(f"Epoch {epoch} | Step {i} | Loss: {loss_val:.4f} | Acc: {acc_val:.2%}")

    return batch_losses, batch_accs, torch.cat(all_preds), torch.cat(all_targets)


def train(lr: float = 0.001, batch_size: int = 64, epochs: int = 5, wandb_enabled: bool = True) -> None:
    """Main training orchestrator."""

    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"learning_rate={lr}",
                f"batch_size={batch_size}",
                f"epochs={epochs}",
                f"wandb.enabled={wandb_enabled}",
                f"optimizer.lr={lr}",
            ],
        )

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    use_wandb = setup_wandb(cfg)

    # Data & Model
    train_set, _ = corrupt_mnist()
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    model = MyAwesomeModel(model_conf=cfg).to(DEVICE)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    loss_fn = instantiate(cfg.loss_fn)

    # Log static samples ONCE
    if use_wandb:
        log_sample_images(train_loader)

    statistics = {"train_loss": [], "train_accuracy": []}

    try:
        logger.info("Starting training...")

        for epoch in range(cfg.epochs):
            # Run Epoch
            losses, accs, epoch_preds, epoch_targets = train_one_epoch(
                model, train_loader, optimizer, loss_fn, epoch, cfg, use_wandb
            )

            # Update Statistics
            statistics["train_loss"].extend(losses)
            statistics["train_accuracy"].extend(accs)

            # End of Epoch Logs
            if use_wandb:
                log_roc_curves(epoch_targets, epoch_preds)

        logger.success("Training complete.")

        # Save Model
        save_path = Path(cfg.logging.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        # Final Metrics & Artifacts
        save_training_plot(statistics, cfg.logging.figure_path)

        final_preds = epoch_preds.argmax(dim=1)
        final_accuracy = accuracy_score(epoch_targets, final_preds)
        final_precision = precision_score(epoch_targets, final_preds, average="weighted")
        final_recall = recall_score(epoch_targets, final_preds, average="weighted")
        final_f1 = f1_score(epoch_targets, final_preds, average="weighted")

        if use_wandb:
            artifact = wandb.Artifact(
                name="corrupt_mnist_model",
                type="model",
                description="A model trained to classify corrupt MNIST images",
                metadata={
                    "accuracy": final_accuracy,
                    "precision": final_precision,
                    "recall": final_recall,
                    "f1": final_f1,
                },
            )
            artifact.add_file(str(save_path))
            wandb.run.log_artifact(artifact)

    except Exception as e:
        logger.exception(f"Training failed: {e}")
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    typer.run(train)
