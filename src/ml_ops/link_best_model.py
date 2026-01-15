# Takes a sweep ID, finds the best run, and links its model artifact to a model registry.

import typer
import wandb
from loguru import logger

app = typer.Typer()


@app.command()
def link_best_model(
    sweep_id: str = typer.Option(..., help="The ID of the sweep to process (entity/project/sweep_id)"),
    registry: str = typer.Option(
        "model-registry/corrupt-mnist", help="The target model registry (<project>/<registry>)"
    ),
) -> None:
    """Links the best model from a WandB sweep to the model registry."""

    registry_path = f"wandb-registry-{registry}"

    # Initialize WandB API
    api = wandb.Api()

    sweep = None
    try:
        logger.info(f"Fetching sweep: {sweep_id}")
        sweep = api.sweep(sweep_id)
    except Exception as e:
        logger.error(f"Could not find sweep or resolve run from ID: {sweep_id}")
        logger.error(f"Details: {e}")
        raise typer.Exit(code=1)

    # Get the best run
    # sweep.best_run() fetches the run with better metric (min/max based on configuration)
    best_run = sweep.best_run()
    if not best_run:
        logger.error("No runs found in sweep or no best run determined.")
        raise typer.Exit(code=1)

    # Attempt to retrieve metric name safely from config
    metric_name = sweep.config.get("metric", {}).get("name", "unknown")
    metric_val = best_run.summary.get(metric_name)

    logger.info(f"Best run found: {best_run.name} (ID: {best_run.id})")
    if metric_name != "unknown":
        logger.info(f"Metric ({metric_name}): {metric_val}")

    # Find the model artifact
    model_artifact = None
    # We iterate over logging artifacts to find the one with type 'model'
    for artifact in best_run.logged_artifacts():
        if artifact.type == "model":
            model_artifact = artifact
            break

    if not model_artifact:
        logger.error(f"No model artifact (type='model') found in run {best_run.id}.")
        raise typer.Exit(code=1)

    logger.info(f"Found artifact: {model_artifact.name}")
    logger.info(f"Linking artifact to registry: {registry_path}...")

    try:
        model_artifact.link(registry_path)
        # or
        # run.link_artifact(model_artifact, registry_path)
        # where `run` is the current WandB run context
        logger.success(f"Successfully linked artifact to {registry_path}")
    except Exception as e:
        logger.error(f"Failed to link artifact: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
