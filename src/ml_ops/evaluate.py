import logging

import torch
from hydra import main
from omegaconf import DictConfig, OmegaConf

from ml_ops.data import corrupt_mnist
from ml_ops.device import DEVICE
from ml_ops.model import MyAwesomeModel

log = logging.getLogger(__name__)


@main(config_path="../../configs", config_name="config", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model.

    Args:
        cfg: Hydra configuration object
    """
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    log.info("Evaluating like my life depended on it")

    model_checkpoint = cfg.get("model_checkpoint", "models/model.pth")
    log.info(f"Checkpoint: {model_checkpoint}")

    model = MyAwesomeModel(model_conf=cfg).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
    accuracy = correct / total
    log.info(f"Test accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate()
