#!/usr/bin/env bash
# Run on the pod once: Python env with CUDA torch matching the local version (pyproject pins the CPU build, so the pod
# uses its own venv and `.venv/bin/python`, never `uv run`, which would re-sync to CPU torch).
set -euo pipefail
cd /workspace/gft
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cuda=${CUDA_TAG:-cu130}   # torch 2.14.1 wheels exist for cu126, cu130, cu132
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python "numpy>=1.26" "scipy>=1.17.1" "matplotlib>=3.8" "pytest>=8"
uv pip install --python .venv/bin/python "torch==2.14.1" --index-url https://download.pytorch.org/whl/$cuda
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nproc
