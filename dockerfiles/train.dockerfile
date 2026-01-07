
###########
# GENERAL #
###########

# Starting from a base image (uv-based image)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install some essentials
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

####################
# PROJECT SPECIFIC #
####################

##
# Copy our project files into the container
#   ->  we only want the essential parts to keep 
#       our Docker image as small as possible

# COPY requirements.txt requirements.txt 
# not needed since we use uv and pyproject.toml
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY data/ data/
COPY configs/ configs/

## 
# Install our project dependencies
# --locked      enforces strict adherence to uv.lock
# --no-cache    disables writing temporary download/wheel files to keep image size small

WORKDIR /

##
# Install dependencies using uv
##

## INEFFICIENT VERSION
## that always rebuilds everything from scratch
# RUN uv sync --locked --no-cache

## OPTIMIZED VERSION
# Below is an optimized version that uses caching of uv downloads/wheels between builds
# to speed up the build process while still keeping the final image small.
# This makes sure we don't always rebuild everything from scratch.
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

##
# Create directories for saving outputs
RUN mkdir -p models reports/figures

##
# The entry point is the application we want to run
# when the container starts up

ENTRYPOINT ["uv", "run", "src/ml_ops/train.py"]

## Building the Docker image:
#
#     docker build -f train.dockerfile . -t train:latest
#
#        -f train.dockerfile <-> f stands for "file" and specifies the Dockerfile to use
#        .                   <-> the build context, i.e. the folder where Docker looks for files
#        -t train:latest     <-> tags the image with the NAME "train" and the TAG "latest"
#
# In general, Docker images are built for a SPECIFIC PLATFORM. 
#
# For example, if you are using a Mac with an M1/M2 chip, then you are running on an ARM architecture. 
# If you are using a Windows or Linux machine, then you are running on an AMD64 architecture. 
# This is important to know when building Docker images. 
# Thus, Docker images you build may not work on platforms different than the ones you build on. 
# You can specify which platform you want to build for by adding the --platform argument to the docker build command:
#
#      docker build --platform linux/amd64 -f train.dockerfile . -t train:latest
#                   --platform linux/arm64
#
# And when running the image
#
#      docker run --platform linux/amd64 train:latest
#                 --platform linux/arm64
#
# These images can take up a lot of space on your machine.
# To clean, use `docker system prune`
# or manually remove them by first `docker images` to list image IDs
# and then `docker rmi IMAGE_ID` to remove specific images.

