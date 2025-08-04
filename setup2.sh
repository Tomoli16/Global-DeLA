#!/bin/bash

conda create -n dela python=3.10 -y
conda activate dela


pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install wandb addict timm "numpy<2.0"

module load CUDA/12.4.0

export CUDA_HOME=/software/genoa/r24.04/CUDA/12.4.0
export TORCH_CUDA_ARCH_LIST="9.0"

conda install -c conda-forge h5py=3.8.0 -y

cd utils/pointnet2_ops_lib/
pip install .

cd ../..

# Remember to install flash attn 
pip install flash-attn --no-build-isolation



echo "Environment setup complete. You can now run DeLA."
# Run with: source setup.sh
