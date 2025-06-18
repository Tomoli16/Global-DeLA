import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import sys
import os

# Projektwurzel bestimmen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(project_root, "modules/mamba"))

from mamba_ssm.modules.mamba2 import Mamba2

# from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath

def build_cu_seqlens(pts, device=None, dtype=torch.long):
    """
    Convert per-scene point counts to cumulative sequence lengths tensor.

    Args:
        pts: List[int] or Tensor [B] of point counts per scene
        device: torch.device for output
        dtype: torch.dtype for output
    Returns:
        cu_seqlens: Tensor [B+1]
    """

    # 1) pts in Tensor umwandeln
    if isinstance(pts, list):
        # dtype must be integer
        pts_tensor = torch.tensor(pts, dtype=torch.long, device=device)
    else:
        pts_tensor = pts.to(device=device, dtype=torch.long)

    # 2) cumulative sequence lengths: [0, N0, N0+N1, ...]
    cu_seqlens = torch.cat([
        pts_tensor.new_zeros(1),        # → [0]
        pts_tensor.cumsum(0)            # → [N0, N0+N1, ...]
    ])  # shape = [B+1]
    return cu_seqlens.to(device=device, dtype=dtype)


class Mamba2Block(nn.Module):
    """
    Block wrapping Mamba2 with pre/post norms, residual and optional drop path.

    Accepts either 3D input [B, L, C] (constant-length) or flat [sum(Ni), C] plus cu_seqlens.
    Automatically handles flattening/unflattening for variable-length sequences.
    """
    def __init__(
            self, 
            dim, 
            layer_idx, 
            drop_path=0.,
            norm_cls=nn.LayerNorm, 
            fused_add_norm=False,
            residual_in_fp32=False, 
            **mamba_kwargs
    ):
        super().__init__()
        # partial erstellt Funktion, bei der bestimmte Argumente bereits gesetzt sind
        self.residual_in_fp32 = residual_in_fp32
        self.fused = fused_add_norm
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        expand = mamba_kwargs.get("expand", 2)
        headdim = (dim*expand) // 8
        self.mamba2 = Mamba2(
            d_model=dim,
            headdim=headdim, 
            expand=expand,
            use_mem_eff_path=False,
            **mamba_kwargs,
        )
        


    def forward(
            self, 
            x, 
            pts=None,
            residual=None, 
            inference_params=None,            
            seq_idx=None, 
            cu_seqlens=None
    ):
        # Pre-Norm + Residual
        if not self.fused:
            res = x if residual is None else residual + self.drop_path(x)
            if self.residual_in_fp32:
                res = res.to(torch.float32)
            x = self.norm(res)
            residual = res
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                x, residual = fused_add_norm_fn(
                    x,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                x, residual = fused_add_norm_fn(
                    self.drop_path(x),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        u = x.unsqueeze(0) 
        cu_seqlens = build_cu_seqlens(
            pts if pts is not None else [x.size(1)] * x.size(0),
            device=u.device
        )
        seq_idx = None
        # Mamba2 forward
        u_out = self.mamba2(
            u,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params
        )


        
        out = u_out.squeeze(0)  # [B, L, C] or [sum(Ni), C]

        return out, residual

