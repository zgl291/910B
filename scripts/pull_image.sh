#!/bin/bash
# =============================================================================
# Pull the CI-built Docker image from GitHub Container Registry
# Run this AFTER the GitHub Actions build completes
# =============================================================================

set -euo pipefail

# Configuration
GHCR_REPO="${GHCR_REPO:-your-username/your-repo}"  # Change to your GitHub repo
IMAGE_TAG="${IMAGE_TAG:-latest}"
OUTPUT_FILE="${OUTPUT_FILE:-ascend-rs-image.tar.gz}"

echo "============================================================"
echo "  Pull Ascend Remote Sensing Container Image"
echo "============================================================"
echo "  Registry: ghcr.io/${GHCR_REPO}"
echo "  Tag:      ${IMAGE_TAG}"
echo "============================================================"

# Check if user is logged in to GHCR
if [ -z "${GH_PAT:-}" ]; then
    echo ""
    echo "You need a GitHub Personal Access Token (PAT) with 'read:packages' scope."
    echo "Create one at: https://github.com/settings/tokens"
    echo ""
    read -p "Enter your GitHub PAT: " GH_PAT
fi

# Login to GHCR
echo ""
echo "Logging in to GHCR..."
echo "${GH_PAT}" | docker login ghcr.io -u "${GITHUB_USER:-$(git config user.name)}" --password-stdin

# Pull the image
echo ""
echo "Pulling image: ghcr.io/${GHCR_REPO}:${IMAGE_TAG}..."
docker pull "ghcr.io/${GHCR_REPO}:${IMAGE_TAG}"

# Optionally save as tarball for offline transfer
echo ""
read -p "Save image as tarball for offline transfer? (y/N): " SAVE
if [[ "${SAVE}" =~ ^[Yy] ]]; then
    echo "Saving image to ${OUTPUT_FILE}..."
    docker save "ghcr.io/${GHCR_REPO}:${IMAGE_TAG}" | gzip > "${OUTPUT_FILE}"
    echo "Saved: $(ls -lh "${OUTPUT_FILE}")"
    echo ""
    echo "To load on another machine:"
    echo "  docker load -i ${OUTPUT_FILE}"
fi

echo ""
echo "============================================================"
echo "  Image ready: ghcr.io/${GHCR_REPO}:${IMAGE_TAG}"
echo "============================================================"
echo ""
echo "Run the container on your 910B server:"
echo "  docker run -it --rm --name ascend-rs \\"
echo "    --device=/dev/davinci0 --device=/dev/davinci_manager \\"
echo "    --device=/dev/devmm_svm --device=/dev/hisi_hdc \\"
echo "    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\"
echo "    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/:ro \\"
echo "    -v \$(pwd)/workspace:/workspace \\"
echo "    --network=host --ipc=host \\"
echo "    ghcr.io/${GHCR_REPO}:${IMAGE_TAG} bash"
echo ""
