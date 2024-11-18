# Use the latest PyTorch base image
FROM pytorch/pytorch:latest

# Install system dependencies in a single RUN command to reduce image layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    less \
    curl \
    wget \
    unzip \
    nano && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN conda install -y \
    pandas \
    jupyter \
    matplotlib \
    seaborn \
    scikit-learn && \
    conda clean --all -y

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    jinja2
