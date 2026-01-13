FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY cloud-req.txt cloud-req.txt
COPY src/cloud_build_artifact_reg/main.py main.py
WORKDIR /
RUN pip install -r cloud-req.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "main.py"]
