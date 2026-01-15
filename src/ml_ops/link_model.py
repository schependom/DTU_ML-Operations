import os

import typer
import wandb


def link_model(artifact_path: str, aliases: list[str] = ["staging"]) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with.

    Example:
        model_management link-model entity/project/artifact_name:version -a staging -a best

    """
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    entity, project, artifact_name_version = artifact_path.split("/")
    artifact_name, _ = artifact_name_version.split(":")

    artifact = api.artifact(artifact_path)

    # If the artifact is already in a registry, use that registry. Otherwise default to 'model-registry'
    target_project = project if "registry" in project else "model-registry"
    target_path = f"{entity}/{target_project}/{artifact_name}"
    print("Linking artifact to:", target_path)

    artifact.link(target_path=target_path, aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
