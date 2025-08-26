from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from torch import nn
import torch

# =====================
# Dataset paths & train
# =====================
# ScanNetV2 dataset path (should contain scans/)
raw_data_path = Path("data/")
processed_data_path = raw_data_path / "scannetv2"
# processed_data_path = Path("")  # Optional override

scan_train = Path(__file__).parent / "scannetv2_train.txt"
scan_val = Path(__file__).parent / "scannetv2_val.txt"
with open(scan_train, 'r') as file:
    scan_train = [line.strip() for line in file.readlines()]
with open(scan_val, 'r') as file:
    scan_val = [line.strip() for line in file.readlines()]

# Training schedule
epoch = 100
warmup = 10
batch_size = 4
learning_rate = 1e-3
label_smoothing = 0.2

# Run Configuration
run_id = "04"  # Default run ID, can be overridden by command line arguments

scan_args = SimpleNamespace()
scan_args.k = [24, 24, 24, 24, 24]
scan_args.grid_size = [0.02, 0.04, 0.08, 0.16, 0.32]
scan_args.max_pts = 80000

scan_warmup_args = deepcopy(scan_args)
scan_warmup_args.grid_size = [0.02, 2, 3.5, 3.5, 4]


# =====================
# DeLA configuration
# =====================
# Group 1: Backbone (shared core model)
backbone = SimpleNamespace(
    # topology
    ks=scan_args.k,
    depths=[4, 4, 4, 8, 4],
    dims=[64, 96, 160, 288, 512],
    grid_size=scan_args.grid_size,
    order=[
        "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
        "hilbert", "hilbert-trans", "z", "z-trans"
    ],
    # heads & channels
    nbr_dims=[32, 32],
    head_dim=288,
    num_classes=20,
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
    cor_std=[1.6, 2.5, 5, 10, 20],
)

def _build_drop_paths(depths, drop_path_max):
    dpr = torch.linspace(0., drop_path_max, sum(depths)).split(depths)
    return [x.tolist() for x in dpr]

def _build_head_drops(num_stages, max_drop=0.2):
    return torch.linspace(0., max_drop, num_stages).tolist()

# Fill derived backbone schedules
backbone.drop_paths = _build_drop_paths(backbone.depths, backbone.drop_path_max)
backbone.head_drops = _build_head_drops(len(backbone.depths), max_drop=0.2)


# Group 2: FlashAttention
flash = SimpleNamespace(
    enabled=False,        # use_flash_attn_blocks
    layers=2,             # flash_attn_layers per block
    num_heads=8,          # flash_num_heads (current model uses 8 internally)
    dropout=0.1,          # flash_dropout
)


# Group 3: Mamba
mamba = SimpleNamespace(
    run=True,               # run_mamba
    depth=[2],              # mamba_depth
    drop_path_rate=0.1,     # mamba_drop_path_rate
    expand=2,               # mamba_expand
    # optional MLP between Mamba blocks
    use_mlp=True,           # mamba_use_mlp
    mlp_ratio=2.0,          # mamba_mlp_ratio
    mlp_act=nn.GELU,        # mamba_mlp_act
)


# -------- Helpers to flatten grouped config into the legacy dela_args --------
def rebuild_dela_args():
    """Flatten grouped config into a single SimpleNamespace for model code."""
    da = SimpleNamespace(**vars(backbone))

    # Flash params (legacy names)
    da.use_flash_attn_blocks = flash.enabled
    da.flash_attn_layers = flash.layers
    da.flash_num_heads = flash.num_heads
    da.flash_dropout = flash.dropout

    # Mamba params (legacy names)
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
    if preset == "default":
        mamba.run = True
        mamba.depth = [2]
        mamba.drop_path_rate = 0.1
        mamba.expand = 2
        mamba.use_mlp = True
    elif preset == "heavy":
        mamba.run = True
        mamba.depth = [4]
        mamba.drop_path_rate = 0.15
        mamba.expand = 2
        mamba.use_mlp = True
    elif preset == "light":
        mamba.run = True
        mamba.depth = [1]
        mamba.drop_path_rate = 0.05
        mamba.expand = 2
        mamba.use_mlp = True
    elif preset == "disabled":
        mamba.run = False
        mamba.depth = [0]
        mamba.drop_path_rate = 0.0
        mamba.expand = 2
        mamba.use_mlp = False
    else:
        raise ValueError(f"Unknown preset: {preset}")
    global dela_args
    dela_args = rebuild_dela_args()


# Backward-compat alias
def configure_mamba2(preset: str = "default"):
    return configure_mamba(preset)


# Apply default presets and build the flat args once
configure_flash_attention("default")     # default to FlashAttention on
configure_mamba("disabled")              # and Mamba off (GDLA-Light)
dela_args = rebuild_dela_args()


# =====================
# Model selection
# =====================
# "dela_semseg", "global_dela", "GDLA-Light", "GDLA-Heavy"
model_type = "GDLA-Light"

# Optional presets bound to model_type for convenience
if model_type == "GDLA-Light":
    configure_flash_attention("default")
    configure_mamba("disabled")
elif model_type == "GDLA-Heavy":
    configure_flash_attention("disabled")
    configure_mamba("default")


# Run Configuration Function
def configure_run(new_run_id: str = "01"):
    global run_id
    run_id = new_run_id