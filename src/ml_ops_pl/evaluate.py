import logging

import torch
from hydra import main
from omegaconf import DictConfig, OmegaConf

from ml_ops_pl.data import corrupt_mnist
from ml_ops_pl.device import DEVICE
from ml_ops_pl.model import MyAwesomeModel

# TODO
