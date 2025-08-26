from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# =====================
# Dataset paths & train
# =====================
# S3DIS dataset path
raw_data_path = Path("data/Stanford3dDataset_v1.2_Aligned_Version")
processed_data_path = raw_data_path.parent / "s3dis"
# If you want to set the processed dataset path, uncomment and set below
# processed_data_path = Path("")

# Training schedule
epoch = 100
warmup = 10
batch_size = 8
learning_rate = 1e-3
label_smoothing = 0.2

# Dataset-specific args
s3dis_args = SimpleNamespace()
s3dis_args.k = [24, 24, 24, 24]
s3dis_args.grid_size = [0.04, 0.08, 0.16, 0.32]
s3dis_args.max_pts = 30000

s3dis_warmup_args = deepcopy(s3dis_args)
s3dis_warmup_args.grid_size = [0.04, 3.5, 3.5, 3.5]


# =====================
# DeLA configuration
# =====================
# Group 1: Backbone (shared core model)
backbone = SimpleNamespace(
    # topology
    ks=s3dis_args.k,
    depths=[4, 4, 8, 4],
    dims=[64, 128, 256, 512],
    grid_size=s3dis_args.grid_size,
    order=[
        "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
        "hilbert", "hilbert-trans", "z", "z-trans"
    ],
    # heads & channels
    nbr_dims=[32, 32],
    head_dim=256,
    num_classes=13,
    # norms & activations
    bn_momentum=0.02,
    act=nn.GELU,
    mlp_ratio=2,
    # regularization
    head_drops=None,      # filled below based on depths
    drop_paths=None,      # filled below based on depths
    drop_path_max=0.1,    # scalar to generate per-layer rates
    # misc
    use_cp=False,
    cor_std=[1.6, 3.2, 6.4, 12.8],
)

def _build_drop_paths(depths, drop_path_max):
    """Build per-stage drop path schedules from a max value."""
    dpr = torch.linspace(0., drop_path_max, sum(depths)).split(depths)
    return [x.tolist() for x in dpr]

def _build_head_drops(num_stages, max_drop=0.15):
    return torch.linspace(0., max_drop, num_stages).tolist()

# Fill derived backbone schedules
backbone.drop_paths = _build_drop_paths(backbone.depths, backbone.drop_path_max)
backbone.head_drops = _build_head_drops(len(backbone.depths), max_drop=0.15)


# Group 2: FlashAttention
flash = SimpleNamespace(
    enabled=False,        # use_flash_attn_blocks
    layers=2,             # flash_attn_layers per block
    num_heads=8,          # flash_num_heads
    dropout=0.1,          # flash_dropout (used in attn + MLP drops inside the block)
    use_order_prompts=False,  # optional order prompts
)


# Group 3: Mamba
mamba = SimpleNamespace(
    run=True,               # run_mamba
    depth=[2],              # mamba_depth (list form for compatibility)
    drop_path_rate=0.1,     # mamba_drop_path_rate
    expand=2,               # mamba_expand
    # Mamba MLP between blocks
    use_mlp=False,          # mamba_use_mlp (default False to match previous preset)
    mlp_ratio=2.0,          # mamba_mlp_ratio
    mlp_act=nn.GELU,        # mamba_mlp_act
)


# -------- Helpers to flatten grouped config into the legacy dela_args --------
def rebuild_dela_args():
    """Flatten grouped config into a single SimpleNamespace for model code."""
    da = SimpleNamespace(**vars(backbone))

    # Flash params (flattened with legacy names)
    da.use_flash_attn_blocks = flash.enabled
    da.flash_attn_layers = flash.layers
    da.flash_num_heads = flash.num_heads
    da.flash_dropout = flash.dropout
    da.use_order_prompts = flash.use_order_prompts

    # Mamba params (flattened with legacy names)
    da.run_mamba = mamba.run
    da.mamba_depth = mamba.depth
    da.mamba_drop_path_rate = mamba.drop_path_rate
    da.mamba_expand = mamba.expand
    da.mamba_use_mlp = mamba.use_mlp
    da.mamba_mlp_ratio = mamba.mlp_ratio
    da.mamba_mlp_act = mamba.mlp_act

    return da


# Presets operating on grouped config, then rebuild flat args
def configure_flash_attention(preset: str = "disabled"):
    if preset == "default":
        flash.enabled = True
        flash.layers = 2
    elif preset == "heavy":
        flash.enabled = True
        flash.layers = 4
    elif preset == "light":
        flash.enabled = True
        flash.layers = 1
    elif preset == "disabled":
        flash.enabled = False
        flash.layers = 0
    else:
        raise ValueError(f"Unknown preset: {preset}")
    global dela_args
    dela_args = rebuild_dela_args()


def configure_mamba(preset: str = "default"):
    """Configure core Mamba settings (not MLP): run, depth, drop_path_rate, expand."""
    if preset == "default":
        mamba.run = True
        mamba.depth = [2]
        mamba.drop_path_rate = 0.1
        mamba.expand = 2
    elif preset == "heavy":
        mamba.run = True
        mamba.depth = [4]
        mamba.drop_path_rate = 0.15
        mamba.expand = 2
    elif preset == "light":
        mamba.run = True
        mamba.depth = [1]
        mamba.drop_path_rate = 0.05
        mamba.expand = 2
    elif preset == "disabled":
        mamba.run = False
        mamba.depth = [0]
        mamba.drop_path_rate = 0.0
        mamba.expand = 2
    else:
        raise ValueError(f"Unknown preset: {preset}")
    global dela_args
    dela_args = rebuild_dela_args()


# Apply default presets and build the flat args once
configure_flash_attention("default")
configure_mamba("disabled")
dela_args = rebuild_dela_args()


# =====================
# Model selection
# =====================
# "dela_semseg", "global_dela", "GDLA-Light", "GDLA-Heavy", or "dela_semseg_baseline"
model_type = "GDLA-Light"

# Optional presets bound to model_type for convenience
# - GDLA-Light: FlashAttention default, Mamba disabled
# - GDLA-Heavy: Mamba default, FlashAttention disabled
if model_type == "GDLA-Light":
    configure_flash_attention("default")
    configure_mamba("disabled")
elif model_type == "GDLA-Heavy":
    configure_flash_attention("disabled")
    configure_mamba("default")
