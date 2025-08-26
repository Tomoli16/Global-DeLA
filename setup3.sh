#!/bin/bash
set -euo pipefail

# --- Conda env ---
conda create -n dela_new python=3.10 -y
conda activate dela_new

# --- PyTorch + vision/audio (alle passend zueinander) ---
pip install -U --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1"

# --- Basics & Libs ---
pip install wandb addict "timm>=0.9.12" "numpy<2.0" scipy

# ChamferDistance (falls du das PyPI-Paket nutzt; lässt CUDA bauen)
pip install chamferdist

# --- pointnet2 ops ---
cd utils/pointnet2_ops_lib/
pip install .
cd ../..

# --- FlashAttention (gegen Torch 2.4.x bauen/holen) ---
pip install --no-build-isolation "flash-attn==2.6.3"

# --- Mamba v2 + causal-conv1d (modern API) ---
cd modules/mamba/
pip install -e ".[causal-conv1d]" --no-build-isolation
cd ../..

echo "Environment setup complete. You can now run DeLA."
echo "Test: python -c \"import torch; import mamba_ssm, causal_conv1d; from flash_attn import flash_attn_interface; print('OK')\""
