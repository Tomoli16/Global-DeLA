from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# S3DIS dataset path
raw_data_path = Path("data/Stanford3dDataset_v1.2_Aligned_Version")

processed_data_path = raw_data_path.parent / "s3dis"
# if you want to set the processed dataset path, uncomment here
#processed_data_path = Path("")

epoch = 100
warmup = 10
batch_size = 8
learning_rate = 1e-3
label_smoothing = 0.2

s3dis_args = SimpleNamespace()
s3dis_args.k = [24, 24, 24, 24]
s3dis_args.grid_size = [0.04, 0.08, 0.16, 0.32]

s3dis_args.max_pts = 30000

s3dis_warmup_args = deepcopy(s3dis_args)
s3dis_warmup_args.grid_size = [0.04, 3.5, 3.5, 3.5]

dela_args = SimpleNamespace()
dela_args.ks = s3dis_args.k
dela_args.depths = [4, 4, 8, 4]
dela_args.grid_size = [0.04, 0.08, 0.16, 0.32]
dela_args.order = [ "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "hilbert", "hilbert-trans", "z", "z-trans" ]

# Order Prompt Configuration - simple True/False setting
dela_args.use_order_prompts = False  # Enable/disable order prompts globally
dela_args.dims = [64, 128, 256, 512]
dela_args.nbr_dims = [32, 32]
dela_args.head_dim = 256
dela_args.num_classes = 13
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
dela_args.head_drops = torch.linspace(0., 0.15, len(dela_args.depths)).tolist()
dela_args.bn_momentum = 0.02
dela_args.act = nn.GELU
dela_args.mlp_ratio = 2
# gradient checkpoint
dela_args.use_cp = False
dela_args.run_mamba = True
dela_args.mamba_depth = [2]  # Mamba2 depth for each stage

# Mamba MLP Configuration
dela_args.mamba_use_mlp = True  # Add MLP between Mamba layers
dela_args.mamba_mlp_ratio = 2.0  # MLP expansion ratio
dela_args.mamba_mlp_act = nn.GELU  # Activation function for Mamba MLPs

dela_args.cor_std = [1.6, 3.2, 6.4, 12.8]

# Flash Attention Block Configuration
dela_args.use_flash_attn_blocks = False  # Enable flash attention blocks in stages
dela_args.flash_attn_layers = 2  # Number of flash attention layers per block

# Model selection: "dela_semseg" or "dela_semseg_attn"
model_type = "dela_semseg_attn2"

# Configuration Presets
def configure_flash_attention(preset="default"):
    """
    Configure flash attention settings with different presets
    
    Args:
        preset (str): Configuration preset
            - "default": Basic flash attention with 2 layers
            - "heavy": More layers for complex scenes  
            - "light": Minimal flash attention for speed
            - "disabled": Traditional LFP only
    """
    if preset == "default":
        dela_args.use_flash_attn_blocks = True
        dela_args.flash_attn_layers = 2
    elif preset == "heavy":
        dela_args.use_flash_attn_blocks = True
        dela_args.flash_attn_layers = 4
    elif preset == "light":
        dela_args.use_flash_attn_blocks = True
        dela_args.flash_attn_layers = 1
    elif preset == "disabled":
        dela_args.use_flash_attn_blocks = False
        dela_args.flash_attn_layers = 0
    else:
        raise ValueError(f"Unknown preset: {preset}")

def configure_mamba_mlp(preset="default"):
    """
    Configure Mamba MLP settings with different presets
    
    Args:
        preset (str): Configuration preset
            - "default": Standard MLP with 2x expansion
            - "heavy": Larger MLP with 4x expansion  
            - "light": Smaller MLP with 1.5x expansion
            - "disabled": No MLP between Mamba layers
    """
    if preset == "default":
        dela_args.mamba_use_mlp = True
        dela_args.mamba_mlp_ratio = 2.0
        dela_args.mamba_mlp_act = nn.GELU
    elif preset == "heavy":
        dela_args.mamba_use_mlp = True
        dela_args.mamba_mlp_ratio = 4.0
        dela_args.mamba_mlp_act = nn.GELU
    elif preset == "light":
        dela_args.mamba_use_mlp = True
        dela_args.mamba_mlp_ratio = 1.5
        dela_args.mamba_mlp_act = nn.ReLU
    elif preset == "disabled":
        dela_args.mamba_use_mlp = False
        dela_args.mamba_mlp_ratio = 2.0
        dela_args.mamba_mlp_act = nn.GELU
    else:
        raise ValueError(f"Unknown preset: {preset}")

# Apply default configuration
configure_flash_attention("disabled")
configure_mamba_mlp("default")
