#!/bin/bash
# =============================================================================
# Run the Ascend 910B container
# Must be run on the 910B server with CANN drivers installed
# =============================================================================
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ascend-remote-sensing}"
IMAGE_TAG="${IMAGE_TAG:-v1.0}"
CONTAINER_NAME="${CONTAINER_NAME:-ascend-rs}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)/workspace}"

echo "============================================================"
echo "  Starting Ascend 910B Container"
echo "============================================================"
echo "  Image:      ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Container:  ${CONTAINER_NAME}"
echo "  Workspace:  ${WORKSPACE_DIR}"
echo "============================================================"

# Create workspace directory if not exists
mkdir -p "${WORKSPACE_DIR}"

# Check if CANN drivers are available
if [ ! -d "/usr/local/Ascend/driver" ]; then
    echo "ERROR: CANN drivers not found at /usr/local/Ascend/driver"
    echo "  Please ensure CANN is installed on the host."
    exit 1
fi

# Check NPU devices
if ! command -v npu-smi &>/dev/null; then
    echo "WARNING: npu-smi not found. NPU monitoring may not work."
    NPU_DEVICES=""
else
    NPU_INFO=$(npu-smi info 2>/dev/null || echo "NPU info unavailable")
    echo ""
    echo "NPU Status:"
    echo "${NPU_INFO}"
    echo ""
fi

# Build docker run command
DOCKER_RUN="docker run -it --rm"

# Container name
DOCKER_RUN="${DOCKER_RUN} --name ${CONTAINER_NAME}"

# NPU device access (Ascend 910B specific)
DOCKER_RUN="${DOCKER_RUN} --device=/dev/davinci0"
DOCKER_RUN="${DOCKER_RUN} --device=/dev/davinci_manager"
DOCKER_RUN="${DOCKER_RUN} --device=/dev/devmm_svm"
DOCKER_RUN="${DOCKER_RUN} --device=/dev/hisi_hdc"

# For multi-card training, expose all NPU devices
# Uncomment if you need all 8 cards:
# for i in $(seq 0 7); do
#     DOCKER_RUN="${DOCKER_RUN} --device=/dev/davinci${i}"
# done

# Mount CANN driver (read-only)
DOCKER_RUN="${DOCKER_RUN} -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
DOCKER_RUN="${DOCKER_RUN} -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/:ro"
DOCKER_RUN="${DOCKER_RUN} -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro"

# Mount workspace
DOCKER_RUN="${DOCKER_RUN} -v ${WORKSPACE_DIR}:/workspace"

# Mount user's data/models if they exist
if [ -d "/data" ]; then
    DOCKER_RUN="${DOCKER_RUN} -v /data:/host_data:ro"
fi

# Network and IPC
DOCKER_RUN="${DOCKER_RUN} --network=host"
DOCKER_RUN="${DOCKER_RUN} --ipc=host"

# Environment variables for NPU
DOCKER_RUN="${DOCKER_RUN} -e ASCEND_OPP_PATH=/usr/local/Ascend/opp"
DOCKER_RUN="${DOCKER_RUN} -e TOOLCHAIN_HOME=/usr/local/Ascend/toolkit"
DOCKER_RUN="${DOCKER_RUN} -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons:\$LD_LIBRARY_PATH"

# Hostname
DOCKER_RUN="${DOCKER_RUN} -h ascend-rs"

# Remove existing container if it exists
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting container..."
echo ""

# Execute
eval exec ${DOCKER_RUN} "${IMAGE_NAME}:${IMAGE_TAG}" bash
