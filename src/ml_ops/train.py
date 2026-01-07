import logging

import matplotlib.pyplot as plt
import torch
from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ml_ops.data import corrupt_mnist
from ml_ops.device import DEVICE
from ml_ops.model import MyAwesomeModel

log = logging.getLogger(__name__)


@main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train a model on MNIST.

    Args:
        cfg: Hydra configuration object
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    log.info("Training day and night")

    model = MyAwesomeModel(model_conf=cfg).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Create optimizer using Hydra instantiate
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % cfg.logging.log_interval == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    log.info("Training complete")
    torch.save(model.state_dict(), cfg.logging.save_path)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(cfg.logging.figure_path)


def main():
    """Entry point for the script."""
    train()


if __name__ == "__main__":
    main()
