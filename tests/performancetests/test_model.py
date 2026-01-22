"""
This script is triggered on a 'staged_model' repository_dispatch event from GitHub.
This kind of event is sent when a model in the WandB registry is staged (i.e., given the 'staged' alias).

The script
    - downloads the staged model from WandB
    - runs a performance test to ensure it meets speed requirements
"""

import os
import time

import torch
import wandb

from ml_ops.model import MyAwesomeModel


def load_model(version):
    """
    Args:
        version (str): The version string (e.g., 'v2') passed via env var MODEL_NAME
    """
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    # Construct the full registry path
    # Structure: entity/project/collection:version
    if "/" in version:
        model_checkpoint = version
    else:
        model_checkpoint = (
            f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/{os.getenv('WANDB_COLLECTION')}:{version}"
        )

    print(f"Downloading artifact: {model_checkpoint}")
    logdir = "."

    # Download the artifact
    try:
        artifact = api.artifact(model_checkpoint)
        # download(root=".") saves files to the current directory
        artifact_dir = artifact.download(root=logdir)
    except wandb.errors.CommError as e:
        raise ValueError(f"Could not fetch artifact {model_checkpoint}. Check your API key and project paths.") from e

    # Locate the model file
    # We assume the artifact contains a single .ckpt file
    file_name = artifact.files()[0].name

    model = MyAwesomeModel()

    return model.load_from_checkpoint(checkpoint_path=f"{logdir}/{file_name}")


def test_model_speed():
    # The YAML sets MODEL_NAME to the version string (e.g., "v2")
    version = os.getenv("MODEL_NAME")
    if not version:
        raise EnvironmentError("MODEL_NAME environment variable is not set")

    model = load_model(version)

    start = time.time()
    for _ in range(100):
        # Ensure input tensor shape matches your model's expectation
        model(torch.rand(1, 1, 28, 28))
    end = time.time()

    # Assert 100 inferences take less than 1 second
    assert end - start < 1, f"Model too slow! Took {end - start:.4f}s"
