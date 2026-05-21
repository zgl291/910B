# =============================================================================
# Huawei Ascend 910B Remote Sensing AI Model Container
# Base: Ascend PyTorch official image (ARM64 + CANN + torch_npu pre-installed)
# Target: Huawei 910B server with Kylin OS
# =============================================================================

ARG CANN_VERSION=8.0.RC2
ARG PYTORCH_IMAGE=ascendai/pytorch:2.1.0-cann${CANN_VERSION}-py3.9

# ---------------------------------------------------------------------------
# Stage 1: Build stage - install all dependencies
# ---------------------------------------------------------------------------
FROM ${PYTORCH_IMAGE} AS builder

LABEL maintainer="your-team"
LABEL description="Remote sensing large model training environment for Ascend 910B"
LABEL version="1.0"

# Prevent interactive prompts during apt/dnf install
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ---- System packages ----
RUN dnf install -y epel-release || yum install -y epel-release || true && \
    dnf install -y --allowerasing \
        cmake \
        ninja-build \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        make \
        perl \
        vim-enhanced \
        openssh-clients \
        openssh-server \
        java-1.8.0-openjdk-devel \
        git \
        wget \
        curl \
        zip \
        unzip \
        bzip2 \
        libffi-devel \
        openssl-devel \
        readline-devel \
        sqlite-devel \
        tk-devel \
        xz-devel \
        zlib-devel \
        && dnf clean all

# ---- Python packages: base dependencies ----
# numpy pinned to 1.23.x or 1.24.0 as per requirements
COPY requirements/base.txt /tmp/requirements/base.txt
RUN pip install --no-cache-dir -r /tmp/requirements/base.txt

# ---- Python packages: image processing ----
COPY requirements/image-processing.txt /tmp/requirements/image-processing.txt
RUN pip install --no-cache-dir -r /tmp/requirements/image-processing.txt

# ---- Python packages: AI frameworks (OpenMMLab ecosystem) ----
COPY requirements/ai-framework.txt /tmp/requirements/ai-framework.txt
RUN pip install --no-cache-dir -r /tmp/requirements/ai-framework.txt

# ---- Python packages: remote sensing & geospatial ----
COPY requirements/remote-sensing.txt /tmp/requirements/remote-sensing.txt
RUN pip install --no-cache-dir -r /tmp/requirements/remote-sensing.txt

# ---- Python packages: distributed training ----
COPY requirements/distributed.txt /tmp/requirements/distributed.txt
RUN pip install --no-cache-dir -r /tmp/requirements/distributed.txt

# ---- Python packages: other scientific computing ----
COPY requirements/scientific.txt /tmp/requirements/scientific.txt
RUN pip install --no-cache-dir -r /tmp/requirements/scientific.txt

# ---- OpenMMLab toolboxes ----
# These need to be installed from source for NPU compatibility
RUN pip install --no-cache-dir \
    'mmdet>=3.0.0' \
    'mmsegmentation>=1.0.0' \
    'mmrotate>=1.0.0' \
    'mmpretrain>=1.0.0' \
    'mmrazor>=1.0.0' \
    'mmdeploy>=1.0.0' \
    'ultralytics>=8.0.0'

# ---- open-CD & BIT-CD (remote sensing change detection) ----
# open-CD from source
RUN cd /opt && \
    git clone https://github.com/likyoo/open-cd.git && \
    cd open-cd && \
    pip install --no-cache-dir -v -e .

# ---- pycocotools ----
RUN pip install --no-cache-dir pycocotools

# ---- thop (FLOPs counter) ----
RUN pip install --no-cache-dir thop

# ---- timm ----
RUN pip install --no-cache-dir timm

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Copy installed packages from builder (if multi-stage) or just verify
# For Ascend images, it's often better to use single-stage due to CANN deps
# The builder stage above serves as a reference; in practice, install directly

# ---- Copy verification scripts ----
COPY scripts/verify_env.py /workspace/scripts/verify_env.py
COPY scripts/quick_test.py /workspace/scripts/quick_test.py

# ---- Create workspace ----
WORKDIR /workspace

# Default command
CMD ["python", "/workspace/scripts/verify_env.py"]
