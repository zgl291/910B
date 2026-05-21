#!/bin/bash
# =============================================================================
# Build the Ascend 910B container image
# Run this script ON THE 910B SERVER (ARM64 + CANN installed)
# =============================================================================
set -euo pipefail

# Configuration
IMAGE_NAME="ascend-remote-sensing"
IMAGE_TAG="v1.0"
CANN_VERSION="${CANN_VERSION:-8.0.RC2}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-ascendai/pytorch:2.1.0-cann${CANN_VERSION}-py3.9}"

echo "============================================================"
echo "  Building Ascend 910B Remote Sensing Container"
echo "============================================================"
echo "  Image:   ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Base:    ${PYTORCH_IMAGE}"
echo "  CANN:    ${CANN_VERSION}"
echo "============================================================"

# Check prerequisites
echo ""
echo "[1/5] Checking prerequisites..."

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Please install docker on the 910B server."
    exit 1
fi

# Check CANN installation (host must have CANN)
if [ ! -d "/usr/local/Ascend" ] && [ ! -d "/usr/local/Ascend/nnae" ]; then
    echo "WARNING: CANN not found at /usr/local/Ascend"
    echo "  The 910B server must have CANN installed for NPU support."
    echo "  Container build will proceed but NPU won't work without CANN."
fi

# Check network connectivity
echo "[2/5] Testing network connectivity..."
if ! ping -c1 -W3 gitee.com &>/dev/null && ! ping -c1 -W3 github.com &>/dev/null; then
    echo "WARNING: No internet connectivity. pip install may fail."
fi

# Pull base image
echo "[3/5] Pulling base image: ${PYTORCH_IMAGE}..."
docker pull "${PYTORCH_IMAGE}" || {
    echo "ERROR: Failed to pull base image ${PYTORCH_IMAGE}"
    echo "  Check network connectivity and image name."
    exit 1
}

# Build the image
echo "[4/5] Building container image..."
docker build \
    --build-arg CANN_VERSION="${CANN_VERSION}" \
    --build-arg PYTORCH_IMAGE="${PYTORCH_IMAGE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile.production \
    .

# Verify the build
echo "[5/5] Verifying build..."
echo ""
docker images "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "============================================================"
echo "  Build Complete!"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "============================================================"
echo ""
echo "Run the container:"
echo "  bash run.sh"
echo ""
echo "Or manually:"
echo "  docker run --rm -it --device=/dev/davinci0 --device=/dev/davinci_manager \\"
echo "    --device=/dev/devmm_svm --device=/dev/hisi_hdc \\"
echo "    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\"
echo "    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \\"
echo "    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\"
echo "    -v \$(pwd)/workspace:/workspace \\"
echo "    --name ascend-rs \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    bash"
echo ""
