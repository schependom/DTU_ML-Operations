import logging
import operator
import os
from typing import Optional

import typer
import wandb
from dotenv import load_dotenv

# Initialize Typer app
app = typer.Typer()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()


@app.command()
def stage_best_model(
    # 1. Source Config
    entity: str = typer.Option(
        ..., envvar="WANDB_ENTITY", help="The WandB entity (user or org) where the project exists."
    ),
    project_name: str = typer.Option(
        ..., envvar="WANDB_PROJECT", help="The specific project where experiments are run."
    ),
    source_artifact: str = typer.Option(
        ..., envvar="MODEL_ARTIFACT_NAME", help="The name of the artifact logged in the project (e.g., 'mnist-cnn')."
    ),
    organization: str = typer.Option(
        ..., envvar="WANDB_ORGANIZATION", help="The WandB organization that owns the target Registry."
    ),
    # 2. Target Registry Config
    target_registry: str = typer.Option("Model-registry", envvar="WANDB_REGISTRY", help="The central Registry name."),
    target_collection: str = typer.Option(
        "corrupt-mnist", envvar="WANDB_COLLECTION", help="The specific Collection inside the Registry."
    ),
    # 3. Promotion Logic Config
    metric_name: str = typer.Option(
        "accuracy", envvar="MODEL_METRIC_NAME", help="The metric key in artifact metadata to optimize."
    ),
    higher_is_better: bool = typer.Option(
        True, envvar="MODEL_METRIC_Maximize", help="True if maximizing metric (accuracy), False if minimizing (loss)."
    ),
):
    """
    Scans a Project's Artifact history, finds the best version, and links it to the Registry.
    """

    # Initialize API
    # WandB automatically picks up WANDB_API_KEY from env vars
    api = wandb.Api()

    # ------------------------------------------------------------------
    # STEP 1: Scan the Source Project
    # ------------------------------------------------------------------
    # Path format: entity/project/artifact_name
    source_path = f"{entity}/{project_name}/{source_artifact}"
    logger.info(f"🔍 Scanning artifact history at: {source_path}")

    try:
        # Get the collection of all versions (v0, v1, v2...)
        artifact_group = api.artifact_type("model", project=project_name).collection(source_artifact)
    except Exception as e:
        logger.error(f"❌ Could not find source artifact '{source_artifact}' in project '{project_name}'.")
        logger.error(f"Details: {e}")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # STEP 2: Find the Best Version
    # ------------------------------------------------------------------
    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_version = None

    # Iterate over every version of this artifact
    for version in artifact_group.artifacts():
        # Check if the metric exists in metadata
        if metric_name in version.metadata:
            val = version.metadata[metric_name]

            # Compare current version against best found so far
            if compare_op(val, best_metric):
                best_metric = val
                best_version = version

    if best_version is None:
        logger.warning(f"⚠️ No artifact versions found with metadata key '{metric_name}'. Cannot promote.")
        raise typer.Exit(code=1)

    logger.info(f"🏆 Best Version Found: {best_version.name} (v{best_version.version})")
    logger.info(f"   Metric ({metric_name}): {best_metric}")

    # ------------------------------------------------------------------
    # STEP 3: Promote to Registry
    # ------------------------------------------------------------------
    # Construct the target path required by WandB Registries.
    # Usually: entity/wandb-registry-{registry_name}/{collection_name}
    target_path = f"{organization}/wandb-registry-{target_registry}/{target_collection}"

    logger.info(f"🚀 Linking to Registry path: {target_path} ...")

    try:
        best_version.link(
            target_path=target_path,
            aliases=["staging", "best-model"],
        )
        logger.info(f"✅ Model successfully staged to Registry Collection: {target_collection}")
    except Exception as e:
        logger.error(f"❌ Failed to link model to registry: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
