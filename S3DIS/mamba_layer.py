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
            expand=4,
            **mamba_kwargs
    ):
        super().__init__()
        # partial erstellt Funktion, bei der bestimmte Argumente bereits gesetzt sind
        self.residual_in_fp32 = residual_in_fp32
        self.fused = fused_add_norm
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        expand = expand
        headdim = (dim*expand) // 8
        self.mamba2 = Mamba2(
            d_model=dim,
            headdim=headdim, # n_heads=channels // 16,
            expand=expand,
            use_mem_eff_path=False,
            **mamba_kwargs,
        )
        self.out_project = nn.Linear(dim*2, dim)  # Output projection to match input dimension


    def forward(
            self, 
            x, 
            pts=None,
            residual=None, 
            inference_params=None,            
            seq_idx=None, 
            cu_seqlens=None,
            bidirectional=True
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
        # Mamba2 forward
        u_out1 = self.mamba2(
            u,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params
        )
        if bidirectional:
            # Reverse the input for backward pass
            u_rev = u.clone()
            for i in range(len(cu_seqlens)-1):
                s, e = cu_seqlens[i].item(), cu_seqlens[i+1].item()
                # print(f"[DEBUG] Szene {i}: slice von {s} bis {e}, Länge = {e-s}")
                seg_before = u[0, s:e, 0].clone()  # Beispiel: erste Dimension, erster Feature-Kanal
                # print("  before:", seg_before[:5], "...", seg_before[-5:])
                
                flipped = u[0, s:e, :].flip(0)     # hier war evtl. Achse vertauscht?
                seg_after = flipped[:, 0]          # erstes Zeit-Element nach Flip
                # print("  flipped first three elements:", seg_after[:3])
                
                u_rev[:, s:e, :] = flipped
                # seg_rev = u_rev[0, s:e, 0]
                # print("  in u_rev:", seg_rev[:5], "...", seg_rev[-5:])

            # Mamba2 backward pass
            u_out_bwd_rev = self.mamba2(
                u_rev,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                inference_params=inference_params
            )
            # Reverse the output of the backward pass
            u_out2 = torch.empty_like(u_out_bwd_rev)
            for i in range(len(cu_seqlens)-1):
                s, e = cu_seqlens[i].item(), cu_seqlens[i+1].item()
                u_out2[:, s:e, :] = u_out_bwd_rev[:, s:e, :].flip(1)
            # Combine the outputs from forward and backward passes
            u_out = torch.cat([u_out1, u_out2], dim=2)  # [1, L, C*2]
            u_out = self.out_project(u_out)  # [1, L, C]
            
        else:
            u_out = u_out1
        
        out = u_out.squeeze(0)  # [B, L, C] or [sum(Ni), C]

        return out, residual

