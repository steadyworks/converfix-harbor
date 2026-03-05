FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y \
    curl wget git vim nano unzip zip p7zip-full \
    python3 python3-pip python3-venv python3-dev python-is-python3 \
    build-essential openssh-server tmux gettext sudo \
    ffmpeg libsm6 libxext6 \
    && pip install jupyter \
    && rm -rf /var/lib/apt/lists/*

# Conda
RUN pip install virtualenv \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda init

# Agent conda environment with ML dependencies
ARG PYTHON_VERSION=3.11
COPY environment/requirements.txt /tmp/requirements.txt
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /opt/conda/bin/conda create -n agent python=${PYTHON_VERSION} -y
RUN /opt/conda/bin/conda run -n agent pip install -r /tmp/requirements.txt && \
    /opt/conda/bin/conda run -n agent pip install tensorflow[and-cuda]==2.17 && \
    /opt/conda/bin/conda run -n agent pip install torch==2.2.0 torchaudio==2.2.0 torchtext==0.17.0 torchvision==0.17.0 && \
    /opt/conda/bin/conda clean -afy
ENV PATH="/opt/conda/bin:${PATH}"

# Setup
COPY environment/entrypoint.sh /entrypoint.sh
COPY environment/instructions.txt /home/instructions.txt
COPY environment/validate_submission.sh /home/validate_submission.sh
RUN useradd -m nonroot \
    && mkdir /home/submission \
    && mkdir -p /home/task_instance \
    && chown nonroot:nonroot /home/submission \
    && chown nonroot:nonroot /tmp \
    && chmod 1777 /tmp \
    && chmod +x /entrypoint.sh
WORKDIR /home

ENTRYPOINT ["/entrypoint.sh"]
