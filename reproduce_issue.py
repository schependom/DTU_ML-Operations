import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to python path to emulate package structure if needed,
# though for config loading it shouldn't matter strictly unless there are custom resolvers.
sys.path.append(os.path.join(os.getcwd(), "src"))


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:
    print("--- Loaded Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    if "wandb" in cfg:
        print(f"cfg.wandb is present.")
        print(f"Type: {type(cfg.wandb)}")
        print(f"Value: {cfg.wandb}")
    else:
        print("cfg.wandb is NOT present.")


if __name__ == "__main__":
    try:
        my_app()
    except Exception as e:
        print(f"Error: {e}")
