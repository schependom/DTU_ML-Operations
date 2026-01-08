FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build tools for compiling any necessary packages
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install wandb package
RUN uv pip install --system wandb

# Copy the tester script
COPY src/ml_ops/wandb_tester.py src/ml_ops/wandb_tester.py

ENTRYPOINT ["python", "src/ml_ops/wandb_tester.py"]
