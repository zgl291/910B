#!/bin/bash
# =============================================================================
# 910B Server Setup Script
# Run this on the bare 910B server BEFORE building the container
# =============================================================================
set -euo pipefail

echo "============================================================"
echo "  Ascend 910B Server Setup"
echo "============================================================"

# ---- 1. Install CANN toolkit ----
CANN_VERSION="${CANN_VERSION:-8.0.RC2}"
CANN_PACKAGE="Ascend-cann-toolkit_${CANN_VERSION}_linux-aarch64.run"

echo ""
echo "[1/4] CANN Toolkit"

if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
    echo "  [OK] CANN already installed"
    ls /usr/local/Ascend/ascend-toolkit/
else
    echo "  Downloading CANN ${CANN_VERSION}..."
    # Download from Huawei mirror
    wget -q "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/${CANN_PACKAGE}" || {
        echo "  [WARN] Download failed. Please download manually from:"
        echo "    https://www.hiascend.com/developer/download/community"
        echo "  Then run: sudo bash ${CANN_PACKAGE} --install"
    }

    if [ -f "${CANN_PACKAGE}" ]; then
        echo "  Installing CANN..."
        sudo bash "${CANN_PACKAGE}" --install
        rm -f "${CANN_PACKAGE}"
    fi
fi

# ---- 2. Install Ascend driver ----
echo ""
echo "[2/4] Ascend Driver"

if [ -d "/usr/local/Ascend/driver" ]; then
    echo "  [OK] Driver already installed"
else
    echo "  [WARN] Please install the Ascend driver from your server vendor"
    echo "    or download from: https://www.hiascend.com/developer/download/community"
fi

# ---- 3. Install Docker ----
echo ""
echo "[3/4] Docker"

if command -v docker &>/dev/null; then
    echo "  [OK] Docker $(docker --version)"
else
    echo "  Installing Docker..."
    # For Kylin OS (based on CentOS/RHEL)
    if command -v yum &>/dev/null || command -v dnf &>/dev/null; then
        curl -fsSL https://get.docker.com | sudo bash -s -- --mirror Aliyun
        sudo systemctl enable docker
        sudo systemctl start docker
    else
        echo "  [ERROR] Unsupported OS. Please install Docker manually."
        exit 1
    fi
fi

# Add current user to docker group
sudo usermod -aG docker "$(whoami)" 2>/dev/null || true

# ---- 4. Verify NPU ----
echo ""
echo "[4/4] NPU Verification"

if command -v npu-smi &>/dev/null; then
    echo "  [OK] npu-smi available"
    echo ""
    npu-smi info
    echo ""
else
    echo "  [WARN] npu-smi not found. Check driver installation."
fi

# Check NPU devices
if ls /dev/davinci* &>/dev/null; then
    echo "  [OK] NPU devices found:"
    ls -la /dev/davinci*
else
    echo "  [WARN] No /dev/davinci* devices found."
    echo "  Check: sudo lsmod | grep davinci"
fi

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Log out and log back in (for docker group to take effect)"
echo "  2. cd to the project directory"
echo "  3. bash build.sh"
echo "  4. bash run.sh"
echo ""
