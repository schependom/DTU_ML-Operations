import logging
import operator
import os
from typing import Optional

import typer
from dotenv import load_dotenv

import wandb

# Initialize Typer app
app = typer.Typer()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (WANDB_API_KEY, WANDB_ENTITY, etc.)
load_dotenv()


@app.command()
def stage_best_model(
    project_name: str = typer.Option(
        ..., help="The specific project where experiments are run (e.g., ml_ops_corrupt_mnist)."
    ),
    source_artifact: str = typer.Option(
        ..., help="The name of the artifact logged in the project (e.g., 'mnist-cnn')."
    ),
    target_registry: str = typer.Option("Model-registry", help="The central Registry name."),
    target_collection: str = typer.Option("corrupt-mnist", help="The specific Collection inside the Registry."),
    metric_name: str = typer.Option("accuracy", help="The metric key in artifact metadata to optimize."),
    higher_is_better: bool = typer.Option(
        True, help="True if maximizing metric (accuracy), False if minimizing (loss)."
    ),
):
    """
    Scans a Project's Artifact history, finds the best version, and links it to the Registry.
    """

    # ------------------------------------------------------------------
    # CONCEPT DEFINITIONS
    # ------------------------------------------------------------------
    # 1. PROJECT: The "Sandbox". Where you run 'wandb.init'.
    #    Contains messy, experimental runs. (Your 'ml_ops_corrupt_mnist')
    #
    # 2. ARTIFACT (Source): A specific output file (model weights) logged
    #    during a run. It has versions (v0, v1, v2).
    #
    # 3. REGISTRY: The "Library". A centralized Organization-level store
    #    for approved models. (Your 'Model-registry')
    #
    # 4. COLLECTION: A folder inside the Registry. It groups versions of
    #    a specific task together. (Your 'corrupt-mnist')
    # ------------------------------------------------------------------

    # 1. Connect to the WandB API
    # We use the entity (organization/user) from env vars, but target the specific project
    entity = os.getenv("WANDB_ENTITY")
    if not entity:
        logger.error("WANDB_ENTITY not found in environment variables.")
        raise typer.Exit(code=1)

    api = wandb.Api()

    # 2. Fetch the Source Artifact "Collection" (The history of versions in the Project)
    # Path format: entity/project/artifact_name
    source_path = f"{entity}/{project_name}/{source_artifact}"
    logger.info(f"Scanning artifact history at: {source_path}")

    try:
        # This gets the container that holds all versions (v0, v1...) of the source artifact
        artifact_group = api.artifact_type("model", project=project_name).collection(source_artifact)
    except Exception as e:
        logger.error(f"Could not find source artifact: {e}")
        raise typer.Exit(code=1)

    # 3. Iterate through versions to find the best one
    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_version = None

    # Note: We iterate over artifact_group.artifacts() which gives us every version (v1, v2, etc.)
    for version in artifact_group.artifacts():
        # IMPORTANT: This assumes you logged metadata WITH the artifact.
        # e.g., run.log_artifact(artifact, metadata={"accuracy": 0.95})
        if metric_name in version.metadata:
            val = version.metadata[metric_name]
            if compare_op(val, best_metric):
                best_metric = val
                best_version = version

    if best_version is None:
        logger.warning(f"No artifact versions found with metadata key '{metric_name}'. Cannot promote.")
        raise typer.Exit(code=1)

    logger.info(f"🏆 Best Version Found: {best_version.name} (v{best_version.version})")
    logger.info(f"   Metric ({metric_name}): {best_metric}")

    # 4. Link to the Registry (The "Promotion" Step)
    # Target Path format: entity/registry-name/collection-name
    target_path = f"wandb-registry-{target_registry}/{target_collection}"

    logger.info(f"Linking to Registry: {target_path} ...")

    best_version.link(
        target_path=target_path,
        aliases=["staging", "best-model"],  # Useful aliases for the registry
    )

    logger.info("✅ Model successfully staged to Registry!")


if __name__ == "__main__":
    app()
