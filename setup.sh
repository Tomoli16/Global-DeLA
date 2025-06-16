# This script sets up the environment for DeLA by installing the required packages.
#!/bin/bash

# conda create -n dela python=3.10 -y
# conda activate dela
# Remember to install mamaba 

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install wandb
pip install addict
pip install timm
# pip install "numpy<2.0"

module load CUDA/12.1.1

export CUDA_HOME=/software/genoa/r24.04/CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="9.0"

conda install -c conda-forge h5py=3.8.0 -y

cd utils/pointnet2_ops_lib/
pip install .


echo "Environment setup complete. You can now run DeLA."
