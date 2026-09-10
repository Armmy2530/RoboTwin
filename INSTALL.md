# RoboTwin Multi-Machine Installation Guide (Pixi + uv + Docker)

This guide provides a reproducible, isolated setup for RoboTwin across different Linux distributions (Ubuntu 20.04/22.04/24.04, Fedora, Arch, Debian) without host compiler conflicts.

---

## Why Pixi + uv?

Robotics and 3D simulation stacks (`sapien`, `warp-lang`, `curobo`) rely heavily on C++ extensions, CUDA kernels, and NumPy C-ABIs. Traditional setup on different host machines often fails due to:
1. **Host GCC mismatch**: Host compilers that are too new (GCC 14/15 on Fedora/Arch/Ubuntu 24.04) or too old (GCC 9 on Ubuntu 20.04) fail to compile CUDA kernels or link with PyTorch C++11 ABI.
2. **Pip dependency resolution speed**: Pip takes minutes or gets stuck resolving CUDA wheels.
3. **NumPy 2.x ABI breakage**: NumPy 2.x broke binary compatibility with pre-compiled C-extensions like CuRobo and PyTorch.

### The Solution:
- **Pixi** isolates the **exact compiler toolchain (GCC 13.4 & G++ 13.4)** and Python 3.10 runtime via `conda-forge`.
- **uv** (bundled inside Pixi) provides blazing-fast, pinned installation of CUDA-enabled PyTorch wheels and robotics Python dependencies (`sapien`, `warp-lang`, `rerun-sdk`, `numpy==1.26.4`).
- **No host contamination**: Everything lives in `.pixi/envs/default`, leaving host packages and Python installations untouched.

---

## System Requirements

- **OS**: Linux (x86_64) — Ubuntu, Fedora, Debian, Arch, RHEL, etc.
- **GPU**: NVIDIA GPU (RTX 20/30/40 series, A-series, H-series) with NVIDIA Driver 535+.
- **CUDA Toolkit**: CUDA 11.8 or 12.x (`nvcc` must be available for building CuRobo).
- **Git & Curl**: Required to fetch repos and tools.

---

## Method 1: Automated 1-Command Setup (Recommended)

Run the included automated setup script directly from `RoboTwin`:

```bash
cd RoboTwin
chmod +x setup_env.sh
./setup_env.sh
```

### What `setup_env.sh` does automatically:
1. Detects GPU, NVIDIA driver, and CUDA toolkit (`nvcc`).
2. Checks for `pixi` (and automatically installs it if missing).
3. Runs `pixi install` to lock GCC 13.4, G++ 13.4, Python 3.10, and ninja.
4. Uses `uv` to install CUDA PyTorch (`torch==2.4.1+cu124 torchvision==0.19.1+cu124`).
5. Applies header compatibility patches if needed.
6. Installs all pinned robotics libraries from `requirements.txt`.
7. Compiles and installs `curobo` using the isolated GCC 13 toolchain.
8. Downloads and extracts simulation assets (`objects`, `embodiments`, `background_texture`) if missing.
9. Runs built-in self-tests verifying PyTorch CUDA, Sapien, CuRobo, Warp, and Rerun.

---

## Method 2: Manual Step-by-Step Setup

If you prefer running commands manually:

### 1. Install Pixi
```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
```

### 2. Initialize Pixi Environment
```bash
cd RoboTwin
pixi install
```

### 3. Install CUDA PyTorch via uv
```bash
pixi run uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1+cu124 torchvision==0.19.1+cu124
```
*(Note: If your machine uses CUDA 11.8, replace `cu124` with `cu118`)*

### 4. Install Dependencies
```bash
pixi run uv pip install -r requirements.txt
```

### 5. Compile CuRobo
```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
pixi run pip install -e envs/curobo --no-build-isolation
```

### 6. Download Assets
```bash
pixi run python assets/_download.py
cd assets
unzip -o -q background_texture.zip
unzip -o -q embodiments.zip
unzip -o -q objects.zip
cd ..
```

---

## Method 3: Clean Docker Container Setup

To test in an isolated Ubuntu 22.04 container or deploy without any local host dependencies:

```bash
cd RoboTwin

# Option A: Automated container test
./test_docker.sh

# Option B: Docker Compose
docker compose build
docker compose run --rm robotwin bash

# Option C: Direct Docker Run
docker build -t robotwin:pixi-uv .
docker run --rm --gpus all \
    -v $(pwd):/workspace/RoboTwin \
    robotwin:pixi-uv \
    pixi run python scripts/capture_camera_frames.py --embodiment yam-abc
```

---

## Verification & Self-Tests

Verify the installation:

```bash
# 1. Verify CUDA & PyTorch
pixi run python -c "import torch; print('CUDA OK:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Verify Simulation & Kinematics Engines
pixi run python -c "import sapien, warp, curobo, rerun; print('Sapien:', sapien.__version__, '| CuRobo:', curobo.__version__)"

# 3. Capture Camera Frames (Verifies Raytracing & Robot Mesh Loading)
pixi run python scripts/capture_camera_frames.py --embodiment yam-abc

# 4. Run Data Collection
bash collect_data.sh put_bottles_dustbin abc_camera_config_arx5 0
```

---

## Convenient Pixi Tasks

Inside `RoboTwin`:
- `pixi run install-deps` : Re-installs/updates dependencies from `requirements.txt`.
- `pixi run build-curobo` : Rebuilds CuRobo with isolated GCC.
- `pixi run download-assets` : Fetches and unzips simulation assets.
- `pixi run test-cam` : Runs headless camera capture test on `yam-abc`.
