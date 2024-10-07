FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    less \
    curl \
    wget \
    unzip \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN conda install pandas jupyter matplotlib seaborn scikit-learn
