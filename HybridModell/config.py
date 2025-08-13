"""
ScanNetV2 Configuration for DeLA with Mamba2 Support

This configuration file supports both Flash Attention and Mamba2 aggregation methods.

Usage Examples:
1. Enable Mamba2 aggregation:
   configure_mamba2("default")  # or "light", "heavy"
   
2. Enable both Flash Attention and Mamba2:
   configure_flash_attention("default")
   configure_mamba2("default")
   
3. Use only traditional LFP (disable all modern methods):
   configure_flash_attention("disabled")
   configure_mamba2("disabled")

4. Set custom run ID:
   configure_run("experiment_01")

Note: Mamba2 requires mamba_ssm package to be installed.
If not available, the model will fall back to linear layers.
"""

from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# ScanNetV2 dataset path
# should contain scans/
raw_data_path = Path("../ScanNetV2/data/")

processed_data_path = raw_data_path / "scannetv2"
# if you want to set the processed dataset path, uncomment here
#processed_data_path = Path("")

scan_train = Path(__file__).parent / "scannetv2_train.txt"
scan_val = Path(__file__).parent / "scannetv2_val.txt"
with open(scan_train, 'r') as file:
    scan_train = [line.strip() for line in file.readlines()]
with open(scan_val, 'r') as file:
    scan_val = [line.strip() for line in file.readlines()]

# Training Configuration
epoch = 100
warmup = 10
batch_size = 16
learning_rate = 1e-3
label_smoothing = 0.2

# Run Configuration
run_id = "01"  # Default run ID, can be overridden by command line arguments

scan_args = SimpleNamespace()
scan_args.k = [24, 24, 24, 24]  # Match S3DIS stages: 4 stages instead of 5
scan_args.grid_size = [0.02, 0.04, 0.08, 0.16]  # Match S3DIS stages: 4 stages instead of 5

scan_args.max_pts = 30000

scan_warmup_args = deepcopy(scan_args)
scan_warmup_args.grid_size = [0.02, 3.5, 3.5, 3.5]  # Match new 4-stage structure

dela_args = SimpleNamespace()
dela_args.pre_training = False
dela_args.dataset_type = "scannetv2"  # Important: Specify dataset type for input dimension compatibility
dela_args.ks = scan_args.k
dela_args.depths = [4, 4, 8, 4]  # Match S3DIS: [4, 4, 8, 4] instead of [4, 4, 4, 8, 4]
dela_args.grid_size = scan_args.grid_size
dela_args.order = [ "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "hilbert", "hilbert-trans", "z", "z-trans" ]
dela_args.dims = [64, 128, 256, 512]  # Match S3DIS exactly: [64, 128, 256, 512]
dela_args.nbr_dims = [32, 32]
dela_args.head_dim = 256  # Match S3DIS: 256 instead of 288
dela_args.num_classes = 20
dela_args.embed_dim = 256  # Match S3DIS: 256 instead of 288 for better transfer compatibility
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
dela_args.head_drops = torch.linspace(0., 0.2, len(dela_args.depths)).tolist()
dela_args.bn_momentum = 0.02
dela_args.act = nn.GELU
dela_args.mlp_ratio = 2
# gradient checkpoint
dela_args.use_cp = False

dela_args.cor_std = [1.6, 3.2, 6.4, 12.8]  # Match S3DIS: 4 stages instead of 5

# Flash Attention Block Configuration
dela_args.use_flash_attn_blocks = False  # Enable flash attention blocks in stages
dela_args.flash_attn_layers = 2  # Number of flash attention layers per block

# Mamba2 Configuration
dela_args.run_mamba = True  # Enable Mamba2 aggregation (set to True to enable)
dela_args.mamba_depth = [2]  # Depth of Mamba2 blocks
dela_args.mamba_drop_path_rate = 0.1  # Drop path rate for Mamba2 blocks
dela_args.dataset_type = "s3dis"

# Model selection: "dela_semseg" or "dela_semseg_attn"
model_type = "dela_semseg_attn"

# Flash Attention Configuration Presets
def configure_flash_attention(preset="disabled"):
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

# Mamba2 Configuration Presets
def configure_mamba2(preset="default"):
    """
    Configure Mamba2 settings with different presets
    
    Args:
        preset (str): Configuration preset
            - "default": Basic Mamba2 with 2 layers
            - "heavy": More layers for complex dependencies
            - "light": Minimal Mamba2 for speed
            - "disabled": No Mamba2 aggregation
    """
    if preset == "default":
        dela_args.run_mamba = True
        dela_args.mamba_depth = [2]
        dela_args.mamba_drop_path_rate = 0.1
    elif preset == "heavy":
        dela_args.run_mamba = True
        dela_args.mamba_depth = [4]
        dela_args.mamba_drop_path_rate = 0.15
    elif preset == "light":
        dela_args.run_mamba = True
        dela_args.mamba_depth = [1]
        dela_args.mamba_drop_path_rate = 0.05
    elif preset == "disabled":
        dela_args.run_mamba = False
        dela_args.mamba_depth = [0]
        dela_args.mamba_drop_path_rate = 0.0
    else:
        raise ValueError(f"Unknown preset: {preset}")

# Apply default configurations
configure_flash_attention("disabled")  # Disabled by default, can be enabled by calling configure_flash_attention("default")
configure_mamba2("default")  # Disabled by default, can be enabled by calling configure_mamba2("default")

# Run Configuration Function
def configure_run(new_run_id="01"):
    """
    Configure run ID for logging and model saving
    
    Args:
        new_run_id (str): Run identifier for this training session
    """
    global run_id
    run_id = new_run_id