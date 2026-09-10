#!/usr/bin/env bash
# ==============================================================================
# RoboTwin Automated Environment Setup Script
# Toolchain: Pixi (GCC 13 + Python 3.10) + uv (CUDA 12.4 PyTorch + Pip)
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

echo "============================================================"
echo "      RoboTwin Environment Setup (Pixi + uv Toolchain)     "
echo "============================================================"

# 1. System & GPU Check
info "Checking NVIDIA GPU and driver..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi not found. Running on CPU-only or inside container without GPU passthrough."
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    info "Found GPU: $GPU_NAME"
fi

# 2. Locate CUDA_HOME & nvcc
info "Checking CUDA Toolkit (nvcc)..."
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
    fi
fi

if [ -n "$CUDA_HOME" ] && [ -f "$CUDA_HOME/bin/nvcc" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    NVCC_VER=$("$CUDA_HOME/bin/nvcc" --version | grep "release" | sed 's/.*release \([0-9\.]*\),.*/\1/')
    success "CUDA Compiler detected: nvcc $NVCC_VER at $CUDA_HOME"
else
    warn "CUDA_HOME / nvcc not detected in standard paths (/usr/local/cuda). CuRobo build will require host nvcc."
fi

# 3. Check / Install Pixi
info "Checking Pixi package manager..."
if ! command -v pixi &> /dev/null; then
    if [ -f "$HOME/.pixi/bin/pixi" ]; then
        export PATH="$HOME/.pixi/bin:$PATH"
    else
        info "Pixi not found. Installing Pixi via official script (https://pixi.sh)..."
        curl -fsSL https://pixi.sh/install.sh | bash
        export PATH="$HOME/.pixi/bin:$PATH"
    fi
fi

if ! command -v pixi &> /dev/null; then
    error "Failed to install or locate pixi. Please visit https://pixi.sh to install manually."
fi
success "Pixi is ready: $(pixi --version)"

# 4. Install Pixi isolated environment (Python 3.10, GCC 13, G++ 13, uv, pip, ninja)
info "Installing Pixi toolchain (isolated GCC 13 + Python 3.10)..."
pixi install
success "Pixi environment locked and initialized at .pixi/envs/default"

# 5. Install PyTorch with CUDA support via uv
info "Installing CUDA-enabled PyTorch 2.4.1 via uv..."
pixi run uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.4.1+cu124" \
    "torchvision==0.19.1+cu124"
success "PyTorch 2.4.1+cu124 installed."

# 6. Apply modern C++ header fix to PyTorch List_inl.h if needed
TORCH_LIST_HDR=".pixi/envs/default/lib/python3.10/site-packages/torch/include/ATen/core/List_inl.h"
if [ -f "$TORCH_LIST_HDR" ]; then
    if grep -q "typename decltype(impl_->list)::difference_type" "$TORCH_LIST_HDR"; then
        info "Patching PyTorch List_inl.h header for modern C++ compatibility..."
        sed -i 's/static_cast<typename decltype(impl_->list)::difference_type>(pos)/pos/g' "$TORCH_LIST_HDR"
        success "PyTorch header patched successfully."
    fi
fi

# 7. Install Core Python Dependencies via uv
info "Installing dependencies from requirements.txt via uv..."
pixi run uv pip install -r requirements.txt
success "Requirements installed."

# 8. Compile and install CuRobo
info "Configuring CuRobo with isolated toolchain..."
if [ -d "envs/curobo" ]; then
    pixi run bash -c '
        export CUDAHOSTCXX="$CONDA_PREFIX/bin/g++"
        export CC="$CONDA_PREFIX/bin/gcc"
        export CXX="$CONDA_PREFIX/bin/g++"
        cd envs/curobo
        python setup.py build_ext --inplace
        pip install -e . --no-build-isolation
    '
    success "CuRobo built and installed."
else
    warn "envs/curobo directory not found; skipping CuRobo build."
fi

# 9. Check Assets
info "Checking RoboTwin assets..."
if [ ! -d "assets/objects" ] || [ ! -d "assets/embodiments" ]; then
    info "Assets not found. Downloading assets from Hugging Face..."
    pixi run python assets/_download.py
    (
        cd assets
        [ -f background_texture.zip ] && unzip -o -q background_texture.zip
        [ -f embodiments.zip ] && unzip -o -q embodiments.zip
        [ -f objects.zip ] && unzip -o -q objects.zip
    )
    success "Assets downloaded and extracted."
else
    success "Assets already present in assets/."
fi

# 10. Verification Self-Tests
echo "============================================================"
info "Running verification self-tests..."
echo "============================================================"

pixi run python -c "
import torch
print('✓ PyTorch Version:', torch.__version__)
print('✓ CUDA Available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ Device Name:    ', torch.cuda.get_device_name(0))
"

pixi run python -c "
import sapien
print('✓ Sapien Version: ', sapien.__version__)
"

pixi run python -c "
import curobo
print('✓ CuRobo Version: ', curobo.__version__)
"

pixi run python -c "
import warp
print('✓ Warp Version:   ', warp.__version__)
"

pixi run python -c "
import rerun
print('✓ Rerun Version:  ', rerun.__version__)
"

echo "============================================================"
success "RoboTwin environment setup is COMPLETE and verified!"
echo "To run scripts in this environment:"
echo "  pixi run python scripts/capture_camera_frames.py --embodiment yam-abc"
echo "Or activate the shell:"
echo "  pixi shell"
echo "============================================================"
