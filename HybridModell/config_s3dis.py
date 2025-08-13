"""
S3DIS Configuration for DeLA HybridModel with Mamba2 Support

This configuration file supports both Flash Attention and Mamba2 aggregation methods
specifically tuned for S3DIS dataset.

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
   configure_run("s3dis_experiment_01")

Note: Mamba2 requires mamba_ssm package to be installed.
If not available, the model will fall back to linear layers.
"""

from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# S3DIS dataset path
raw_data_path = Path("../S3DIS/data/Stanford3dDataset_v1.2_Aligned_Version")
processed_data_path = raw_data_path.parent / "s3dis"

# Training Configuration
epoch = 100
warmup = 10
batch_size = 8
learning_rate = 1e-3
label_smoothing = 0.2

# Run Configuration
run_id = "s3dis_finetune_01"  # Default run ID, can be overridden by command line arguments

# S3DIS specific args
s3dis_args = SimpleNamespace()
s3dis_args.k = [24, 24, 24, 24]
s3dis_args.grid_size = [0.04, 0.08, 0.16, 0.32]
s3dis_args.max_pts = 30000

s3dis_warmup_args = deepcopy(s3dis_args)
s3dis_warmup_args.grid_size = [0.04, 3.5, 3.5, 3.5]

# DeLA args für S3DIS (13 Klassen)
dela_args = SimpleNamespace()
dela_args.pre_training = False
dela_args.dataset_type = "s3dis"  # Important: Specify dataset type for input dimension compatibility
dela_args.ks = s3dis_args.k
dela_args.depths = [4, 4, 8, 4]
dela_args.grid_size = [0.04, 0.08, 0.16, 0.32]
dela_args.order = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "hilbert", "hilbert-trans", "z", "z-trans"]
dela_args.dims = [64, 128, 256, 512]
dela_args.nbr_dims = [32, 32]
dela_args.head_dim = 256
dela_args.num_classes = 13  # S3DIS has 13 classes
dela_args.embed_dim = 256  # Embedding dimension for pre-training consistency

# Drop path configuration
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(dela_args.depths)).split(dela_args.depths)
dela_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
dela_args.head_drops = torch.linspace(0., 0.15, len(dela_args.depths)).tolist()

# Model configuration
dela_args.bn_momentum = 0.02
dela_args.act = nn.GELU
dela_args.mlp_ratio = 2
dela_args.use_cp = False  # Gradient checkpoint
dela_args.cor_std = [1.6, 3.2, 6.4, 12.8]

# Flash Attention Block Configuration
dela_args.use_flash_attn_blocks = True  # Enable flash attention blocks in stages
dela_args.flash_attn_layers = 2  # Number of flash attention layers per block

# Mamba2 Configuration
dela_args.run_mamba = True  # Enable Mamba2 aggregation
dela_args.mamba_depth = [4]  # Depth of Mamba2 blocks
dela_args.mamba_drop_path_rate = 0.1  # Drop path rate for Mamba2 blocks

# Mamba MLP Configuration
dela_args.mamba_use_mlp = True  # Add MLP between Mamba layers
dela_args.mamba_mlp_ratio = 2.0  # MLP expansion ratio
dela_args.mamba_mlp_act = nn.GELU  # Activation function for Mamba MLPs

# Model selection
model_type = "hybrid"  # Use HybridModel

# Flash Attention Configuration Presets
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

# Mamba MLP Configuration Presets
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

# Apply default configurations
configure_flash_attention("disabled")
configure_mamba2("default")
configure_mamba_mlp("disabled")

# Run Configuration Function
def configure_run(new_run_id="s3dis_finetune_01"):
    """
    Configure run ID for logging and model saving
    
    Args:
        new_run_id (str): Run identifier for this training session
    """
    global run_id
    run_id = new_run_id

# S3DIS class names for reference
S3DIS_CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]

# S3DIS class weights (if needed for weighted loss)
S3DIS_CLASS_WEIGHTS = None  # Can be set based on class frequencies

# Validation configuration
val_args = SimpleNamespace()
val_args.partition = "5"  # Default validation on Area 5
val_args.test_areas = ["1", "2", "3", "4", "5", "6"]  # All areas for testing

#python finetune.py --pretrained_path output/model/pretrain_01/best.pt --freeze_encoder --run_id s3dis_decoder_only
