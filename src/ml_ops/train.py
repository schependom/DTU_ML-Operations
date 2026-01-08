import os
from pathlib import Path
from typing import Any, Dict, cast

import hydra
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from ml_ops.data import corrupt_mnist
from ml_ops.device import DEVICE
from ml_ops.model import MyAwesomeModel

# Load environment variables immediately
load_dotenv()


def setup_wandb(cfg: DictConfig) -> bool:
    """Initializes Weights & Biases based on configuration and env vars.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        bool: True if WandB was successfully initialized, False otherwise.
    """
    if not cfg.wandb.enabled:
        logger.info("WandB disabled via config.")
        return False

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY not found in .env. Skipping WandB.")
        return False

    try:
        wandb.login(key=api_key)
        config_dict = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            config=config_dict,
        )
        logger.success("Successfully initialized Weights & Biases.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}. Proceeding without it.")
        return False


def save_plot(statistics: dict, save_path: str) -> None:
    """Generates and saves training plots.

    Args:
        statistics: Dictionary containing loss and accuracy lists.
        save_path: Destination path for the figure.
    """
    try:
        # Use non-interactive backend to avoid issues on headless servers
        plt.switch_backend("agg")
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].plot(statistics["train_loss"])
        axs[0].set_title("Train loss")
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Loss")

        axs[1].plot(statistics["train_accuracy"])
        axs[1].set_title("Train accuracy")
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Accuracy")

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Training plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Main training pipeline.

    Args:
        cfg: Hydra configuration object.
    """
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Using device: {DEVICE}")

    # 1. Setup WandB
    use_wandb = setup_wandb(cfg)

    # 2. Data & Model Setup
    model = MyAwesomeModel(model_conf=cfg).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)

    loss_fn = instantiate(cfg.loss_fn)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # 3. Training Loop
    statistics = {"train_loss": [], "train_accuracy": []}

    logger.info("Starting training...")

    try:
        for epoch in range(cfg.epochs):
            model.train()
            running_loss = 0.0

            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

                # Metrics
                current_loss = loss.item()
                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()

                statistics["train_loss"].append(current_loss)
                statistics["train_accuracy"].append(accuracy)

                if use_wandb:
                    wandb.log({"train_loss": current_loss, "train_accuracy": accuracy, "epoch": epoch})

                if i % cfg.logging.log_interval == 0:
                    logger.info(f"Epoch {epoch} | Iter {i} | Loss: {current_loss:.4f} | Acc: {accuracy:.2%}")

        logger.success("Training complete.")

        # 4. Save Artifacts
        save_path = Path(cfg.logging.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        save_plot(statistics, cfg.logging.figure_path)

    except Exception as e:
        logger.exception(f"Training interrupted: {e}")
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    train()
