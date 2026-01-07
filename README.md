# DTU Machine Learning Operations

Repository for the DTU Machine Learning Operations course.

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Installation and notes on using `uv`

### Virtual environment

Activate the virtual environment with:

```bash
source .venv/bin/activate
```

Install dependencies with:

```bash
uv sync
```

Install an optional dependency group with:

```bash
uv sync --group <group-name>
```

Install all dependency groups with:

```bash
uv sync --all-groups
```

### Dependency management

Add a new dependency with (or add it straight to `pyproject.toml` and run `uv sync`):

```bash
uv add <package-name>
# e.g. uv add numpy
```

To add a development dependency, use:

```bash
uv add <package-name> --group dev
# e.g. uv add pytest --group dev
```

### Requirements files

The `requirements.txt` and `requirements-dev.txt` files are automatically generated from `pyproject.toml` and `uv.lock`. They are provided for compatibility with tools that do not support `pyproject.toml` (e.g. some CIs or older docker builds), and for users who prefer `pip` + `conda`.

Do **not** edit these files manually. Instead, update `pyproject.toml` and then run the following command to sync them:

```bash
uv export --format requirements-txt --output-file requirements.txt
uv export --format requirements-txt --only-group dev --output-file requirements-dev.txt
```

### Running scripts

Running a script inside the virtual environment can be done with:

```bash
uv run <script-name>.py
# e.g. uv run src/ml_ops/train.py
```

This can get quite tedious, so we can make an alias `uvr` for this:

```bash
echo "alias uvr='uv run'" >> ~/.bashrc
source ~/.bashrc
```

Now you can run scripts like this:

```bash
uvr script.py
# e.g. uvr src/ml_ops/train.py
```

### Other `uv` commands

Change Python version with:

```bash
uv python pin <version>
```

### Using `uvx` for global tools

`uvx` can be used to install _tools_ (which are external command line tools, not libraries used in your code) globally on your machine. Tools include `black`, `ruff`, `pytest` or the simple `cowsay`. You can install such tools with `uvx`. For example:

```bash
uvx add cowsay
```

Then you can run the tool like this:

```bash
uvx cowsay -t "muuh"
```

If you run above command without having installed `cowsay` with `uvx`, it will install it for you automatically.

## Usage

### Version control

Clone the repo:

```bash
git clone git@github.com:schependom/DTU_ML-Operations.git
cd DTU_ML-Operations
```

Authenticate DVC using SSH (make sure you have access to the remote):

```bash
dvc remote modify --local myremote auth ssh
```

Pull data from DVC remote:

```bash
dvc pull
```

You can use `invoke` to run common tasks. To list available tasks, run:

```bash
invoke --list
# Available tasks:
#
#   build-docs        Build documentation.
#   docker-build      Build docker images.
#   preprocess-data   Preprocess data.
#   serve-docs        Serve documentation.
#   test              Run tests.
#   train             Train model.
```

Now, to run a task, use:

```bash
invoke <task-name>
# e.g. invoke preprocess-data
```

After preprocessing data (v2), you can push (Data Version Control [DVC]) changes to the remote with:

```bash
dvc add data
git add data.dvc # or `git add .`
git commit -m "Add new data"
git tag -a v2.0 -m "Version 2.0"
# Why tag? To mark a specific point in git history as important (e.g., a release)
#   -a to create an annotated tag
#   -m to add a message to the tag
dvc push
git push origin main --tags
```

Or simply use:

```bash
invoke dvc --folder 'data' --message 'Add new data'
```

To go back to a specific version later, you can checkout the git tag:

```bash
git switch v1.0 # or `git checkout v1.0`
dvc checkout
```

To go back to the latest version, use:

```bash
git switch main # or `git checkout main`
dvc checkout
```

### Training

To train the model using the default configuration (`configs/config.yaml`), run:

```bash
uvr invoke train
uvr src/ml_ops/train.py
uvr train # because we configured a script entry point in pyproject.toml
```

To train with custom parameters:

```bash
# Switch to Nesterov SGD
uvr src/ml_ops/train.py optimizer=nesterov

# Override optimizer params
uvr src/ml_ops/train.py optimizer=adam optimizer.lr=0.01
```

Or create a new config file `configs/custom_config.yaml` with:

```yaml
defaults:
    - my_new_model_conf
    - my_new_training_conf
    - optimizer: my_preferred_optimizer
    - _self_
use_my_new_model_conf: true
use_my_new_training_conf: true
```

Then run training with the new config:

```bash
uvr src/ml_ops/train.py --config-name=custom_config
```

## Containerization

Docker containers provide isolated, reproducible environments for training and evaluating models. This project includes optimized Dockerfiles for both operations.

### Building Docker Images

Build the training image:

```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
```

Build the evaluation image:

```bash
docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
```

<details>
<summary>Cross-platform builds (e.g., building for AMD64 on ARM Mac)</summary>

Some systems (like Apple M1/M2 Macs) use ARM architecture, which can lead to compatibility issues when sharing Docker images with others using AMD64 architecture (common in cloud and many desktops). To ensure compatibility, you can build images for a specific platform using the `--platform` flag.

```bash
# ARM Mac (Apple Silicon) -> AMD64
docker build --platform linux/amd64 -f dockerfiles/train.dockerfile . -t train:latest

# Windows on AMD64 -> ARM64 (e.g. Apple Silicon)
docker build --platform linux/arm64 -f dockerfiles/train.dockerfile . -t train:latest
```

</details>

### Running Docker Containers

Run training (using configurations in `configs/`):

```bash
docker run --name experiment1 train:latest
```

Run training with custom parameters:

```bash
docker run --name experiment2 train:latest --lr 0.01 --epochs 20
```

Run evaluation (requires model file):

```bash
docker run --name eval1 evaluate:latest models/model.pth
```

Run evaluation with mounted volumes (recommended - keeps data and models on host):

```bash
# Mount model and data directories
docker run --name eval1 --rm \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  evaluate:latest \
  /models/model.pth

# Or mount specific files
docker run --name eval1 --rm \
  -v $(pwd)/models/model.pth:/models/model.pth \
  -v $(pwd)/data/processed/test_images.pt:/data/processed/test_images.pt \
  -v $(pwd)/data/processed/test_target.pt:/data/processed/test_target.pt \
  evaluate:latest \
  /models/model.pth
```

### Interactive Mode

Debug or explore the container interactively:

```bash
docker run --rm -it --entrypoint sh train:latest
```

### Copying Files from Container

After training, copy outputs from container to host:

```bash
# Trained model
docker cp experiment1:/models/model.pth models/model.pth
# Training statistics figure
docker cp experiment1:/reports/figures/training_statistics.png reports/figures/training_statistics.png
```

### Container and Image Management

#### Containers

List all **containers** (running and stopped):

```bash
docker ps -a
# or `docker container ls -a`
```

Remove a specific container:

```bash
docker rm experiment1
```

Clean up stopped containers:

```bash
docker container prune
```

#### Images

List all **images**:

```bash
docker images
```

Remove a specific image (only if you want to rebuild or no longer need it):

```bash
docker rmi train:latest
```

Clean up dangling images (unnamed `<none>` images from rebuilds):

```bash
docker image prune
```

#### System-wide Cleanup

Clean up everything (stopped containers, dangling images, unused networks):

```bash
docker system prune
```

### Using Volumes for Persistent Storage

A more efficient strategy is to mount a **volume** that is shared between the host and the container using the `-v` option. This allows you to:

-   Access outputs without copying files from the container
-   Provide inputs (data, models) without building them into the image
-   Persist data after the container is removed

Example - automatically save trained model to host:

```bash
docker run --name experiment3 --rm -v $(pwd)/models:/models train:latest
```

The model will be saved directly to your local `models/` directory.

### Docker Best Practices

-   **Use `--rm`**: Automatically remove containers after they exit to avoid clutter
-   **Mount volumes**: For data and outputs instead of copying files
-   **Use `.dockerignore`**: Exclude unnecessary files from build context for faster builds
-   **Name your containers**: Makes them easier to reference with `--name`
-   **Tag images properly**: Use meaningful tags beyond `latest` for versioning
