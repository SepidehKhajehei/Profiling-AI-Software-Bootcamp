#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Brev (VM mode) setup for the Profiling AI Software Bootcamp, Labs 1-3.
# Target: one instance per participant with 4x A100 GPUs, single node.
#
# Safe to re-run. Works whether it is executed as root or as the default user,
# and whether or not Brev has already cloned the repository.
#
# Optional environment variables (can be set as Launchable launch parameters):
#   REPO_URL       git URL to clone if the repo is not found on the instance
#   REPO_BRANCH    branch to clone (default: main)
#   JUPYTER_PORT   port for the JupyterLab service (default: 8888)
#   JUPYTER_IP     bind address (default: 127.0.0.1, reached through a Brev secure link/tunnel)
#   JUPYTER_TOKEN  JupyterLab token (default: empty; set one if JUPYTER_IP is 0.0.0.0)
#   DATA_ZIP_URL   mirror for data-list.zip (default: Google Drive via download-data.py)
#   MODELS_ZIP_URL mirror for saved_models.zip

set -euo pipefail

DATA_ZIP_URL="${DATA_ZIP_URL:-}"
MODELS_ZIP_URL="${MODELS_ZIP_URL:-}"

REPO_URL="${REPO_URL:-https://github.com/openhackathons-org/Profiling-AI-Software-Bootcamp.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_IP="${JUPYTER_IP:-127.0.0.1}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
KERNEL_NAME="profiling-bootcamp"
KERNEL_DISPLAY="Profiling Bootcamp (Python 3.12)"

# ---------------------------------------------------------------------------
# Users and privileges
# ---------------------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
    if id ubuntu >/dev/null 2>&1; then LAB_USER=ubuntu; else LAB_USER=root; fi
else
    SUDO="sudo"
    LAB_USER="$(id -un)"
fi
LAB_HOME="$(getent passwd "$LAB_USER" | cut -d: -f6)"

LOG="$LAB_HOME/brev-bootcamp-setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "==== $(date) : bootcamp setup (user=$LAB_USER) ===="

as_user() {
    if [ "$(id -un)" = "$LAB_USER" ]; then
        bash -c "$1"
    else
        sudo -u "$LAB_USER" -H bash -c "$1"
    fi
}

warn() { echo "WARNING: $*"; WARNINGS+=("$*"); }
WARNINGS=()

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "--> Installing system packages"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends git unzip wget curl ca-certificates

# ---------------------------------------------------------------------------
# 2. Locate (or clone) the repository
# ---------------------------------------------------------------------------
SCRIPT_DIR=""
[ -n "${BASH_SOURCE[0]:-}" ] && SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
REPO_DIR=""
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../requirements.txt" ] && [ -d "$SCRIPT_DIR/../workspace" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    found="$(find "$LAB_HOME" -maxdepth 4 -type f -name start_here.ipynb -path '*Profiling-AI-Software-Bootcamp/workspace/*' 2>/dev/null | head -1)"
    if [ -n "$found" ]; then REPO_DIR="$(dirname "$(dirname "$found")")"; fi
fi
if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$LAB_HOME/Profiling-AI-Software-Bootcamp"
    echo "--> Cloning $REPO_URL ($REPO_BRANCH) into $REPO_DIR"
    as_user "git clone --depth 1 --branch '$REPO_BRANCH' '$REPO_URL' '$REPO_DIR'"
fi
WORKSPACE="$REPO_DIR/workspace"
VENV="$REPO_DIR/.venv"
echo "--> Repository: $REPO_DIR"
# Brev clones the default branch; switch to the branch that has the brev/ folder
if [ ! -f "$REPO_DIR/brev/adapt_notebooks.py" ]; then
    echo "--> brev/ not found; checking out $REPO_BRANCH from $REPO_URL"
    as_user "git -C '$REPO_DIR' fetch --depth 1 '$REPO_URL' '$REPO_BRANCH' && git -C '$REPO_DIR' checkout -B '$REPO_BRANCH' FETCH_HEAD" || true
fi
if [ ! -f "$REPO_DIR/brev/adapt_notebooks.py" ]; then
    echo "ERROR: $REPO_DIR has no brev/ folder. Set REPO_URL/REPO_BRANCH to a repo/branch that contains it."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Nsight Systems CLI (latest from the NVIDIA CUDA apt repository)
# ---------------------------------------------------------------------------
echo "--> Installing Nsight Systems"
. /etc/os-release
DISTRO="ubuntu${VERSION_ID//./}"
case "$(dpkg --print-architecture)" in
    amd64) CUDA_ARCH=x86_64 ;;
    arm64) CUDA_ARCH=sbsa ;;
    *) CUDA_ARCH="$(dpkg --print-architecture)" ;;
esac
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    wget -q -O /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb"
    $SUDO dpkg -i /tmp/cuda-keyring.deb
    $SUDO apt-get update -y
fi
NSYS_PKG="$(apt-cache search --names-only '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -1)"
if [ -n "$NSYS_PKG" ]; then
    $SUDO apt-get install -y --no-install-recommends "$NSYS_PKG"
else
    warn "No nsight-systems package found in the CUDA apt repo for $DISTRO"
fi
NSYS_BIN="$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -V | tail -1 || true)"
[ -z "$NSYS_BIN" ] && NSYS_BIN="$(command -v nsys || true)"
[ -z "$NSYS_BIN" ] && warn "nsys not found; profiling cells will fail"

# ---------------------------------------------------------------------------
# 4. Allow GPU metrics sampling (nsys --gpu-metrics-devices) for non-root users
# ---------------------------------------------------------------------------
echo "--> Enabling GPU performance counters for non-admin users"
profiling_restricted() {
    grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params 2>/dev/null
}
if profiling_restricted; then
    echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | $SUDO tee /etc/modprobe.d/nvidia-profiling.conf >/dev/null
    command -v update-initramfs >/dev/null && $SUDO update-initramfs -u -k all >/dev/null 2>&1 || true
    # Reload the driver now so a reboot is not needed (only works while no process uses the GPUs)
    LOADED_MODULES="$(lsmod | awk '/^nvidia/ {print $1}')"
    # Fabric Manager (NVSwitch systems) and persistenced keep the driver busy
    $SUDO systemctl stop nvidia-fabricmanager nvidia-persistenced 2>/dev/null || true
    $SUDO modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
    for m in nvidia $LOADED_MODULES; do $SUDO modprobe "$m" 2>/dev/null || true; done
    $SUDO systemctl start nvidia-persistenced 2>/dev/null || true
    systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1 \
        && $SUDO systemctl start nvidia-fabricmanager 2>/dev/null || true
    if profiling_restricted; then
        warn "GPU counters are still admin-only. Reboot the instance once (sudo reboot) to apply."
    fi
fi

# ---------------------------------------------------------------------------
# 5. Python environment (uv + Python 3.12, as in Deployment_Guide.md)
# ---------------------------------------------------------------------------
echo "--> Creating Python environment"
as_user 'command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh'
UV="$LAB_HOME/.local/bin/uv"
[ -x "$UV" ] || UV="$(as_user 'command -v uv')"
as_user "cd '$REPO_DIR' && '$UV' venv --allow-existing --python 3.12 '$VENV' \
    && '$UV' pip install --python '$VENV/bin/python' -r requirements.txt ipykernel"

# Make the newest nsys win over any older one on PATH inside the lab environment
[ -n "$NSYS_BIN" ] && as_user "ln -sf '$NSYS_BIN' '$VENV/bin/nsys'"

# ---------------------------------------------------------------------------
# 6. Datasets and saved models
# ---------------------------------------------------------------------------
echo "--> Downloading datasets"
# Mirrors avoid Google Drive quota errors when many instances download at once
[ -n "$DATA_ZIP_URL" ] && [ ! -f "$WORKSPACE/data/data-list.zip" ] \
    && as_user "wget -q -O '$WORKSPACE/data/data-list.zip' '$DATA_ZIP_URL'"
[ -n "$MODELS_ZIP_URL" ] && [ ! -f "$WORKSPACE/source_code/saved_models.zip" ] \
    && as_user "wget -q -O '$WORKSPACE/source_code/saved_models.zip' '$MODELS_ZIP_URL'"
if [ ! -f "$WORKSPACE/data/data-list.zip" ] || [ ! -f "$WORKSPACE/source_code/saved_models.zip" ]; then
    as_user "cd '$WORKSPACE' && '$VENV/bin/python' source_code/download-data.py" \
        || warn "Google Drive download failed (quota?). Re-run: cd $WORKSPACE && $VENV/bin/python source_code/download-data.py"
fi
[ -f "$WORKSPACE/data/data-list.zip" ] && as_user "unzip -q -u '$WORKSPACE/data/data-list.zip' -d '$WORKSPACE/data/'"
[ -f "$WORKSPACE/source_code/saved_models.zip" ] && as_user "unzip -q -u '$WORKSPACE/source_code/saved_models.zip' -d '$WORKSPACE/source_code/'"

# dp/ and mp/ download MNIST from every rank at once; fetch it up front to avoid the race
as_user "cd '$WORKSPACE/source_code' && '$VENV/bin/python' -c \"
from torchvision import datasets
datasets.MNIST('../data/mnist_data', download=True, train=True)
datasets.MNIST('../data/mnist_data', download=True, train=False)
\"" || warn "MNIST pre-download failed"

# ---------------------------------------------------------------------------
# 7. Adapt notebooks for a single 4-GPU node and register the Jupyter kernel
# ---------------------------------------------------------------------------
echo "--> Adapting notebooks"
as_user "'$VENV/bin/python' '$REPO_DIR/brev/adapt_notebooks.py' '$WORKSPACE'"

LAB_PATH="$VENV/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
as_user "'$VENV/bin/python' -m ipykernel install --user --name '$KERNEL_NAME' --display-name '$KERNEL_DISPLAY'"
as_user "'$VENV/bin/python' - <<'EOF'
import json, os
path = os.path.expanduser('~/.local/share/jupyter/kernels/$KERNEL_NAME/kernel.json')
spec = json.load(open(path))
spec['env'] = {'PATH': '$LAB_PATH', 'VIRTUAL_ENV': '$VENV'}
json.dump(spec, open(path, 'w'), indent=1)
EOF"

# ---------------------------------------------------------------------------
# 8. JupyterLab service (survives reboots; the Brev setup script does not re-run)
# ---------------------------------------------------------------------------
echo "--> Starting JupyterLab on $JUPYTER_IP:$JUPYTER_PORT"
SERVICE=/etc/systemd/system/bootcamp-jupyter.service
$SUDO tee "$SERVICE" >/dev/null <<EOF
[Unit]
Description=Profiling AI Software Bootcamp JupyterLab
After=network-online.target

[Service]
User=$LAB_USER
WorkingDirectory=$WORKSPACE
Environment=PATH=$LAB_PATH
Environment=VIRTUAL_ENV=$VENV
ExecStart=$VENV/bin/jupyter-lab --no-browser --ip=$JUPYTER_IP --port=$JUPYTER_PORT --ServerApp.token=$JUPYTER_TOKEN --ServerApp.allow_origin=* --notebook-dir=$WORKSPACE
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
if ss -ltn 2>/dev/null | grep -q ":$JUPYTER_PORT " && ! systemctl is-active --quiet bootcamp-jupyter; then
    warn "Port $JUPYTER_PORT is already in use (Brev Jupyter?). Either disable Brev's Jupyter, set JUPYTER_PORT, or use the existing Jupyter with the '$KERNEL_DISPLAY' kernel."
else
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable bootcamp-jupyter >/dev/null
    $SUDO systemctl restart bootcamp-jupyter
fi

# ---------------------------------------------------------------------------
# 9. Verify
# ---------------------------------------------------------------------------
echo "--> Verifying"
GPU_COUNT="$(nvidia-smi -L | wc -l)"
nvidia-smi -L
[ "$GPU_COUNT" -ge 4 ] || warn "Found $GPU_COUNT GPUs; the labs need 4"
[ -n "$NSYS_BIN" ] && "$NSYS_BIN" --version
as_user "'$VENV/bin/python' -c 'import torch; print(\"torch\", torch.__version__, \"cuda devices:\", torch.cuda.device_count())'"
ls "$WORKSPACE/data"

echo
echo "==== Setup finished. Log: $LOG ===="
echo "JupyterLab: port $JUPYTER_PORT, open start_here.ipynb"
for w in "${WARNINGS[@]}"; do echo "WARNING: $w"; done
