import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import sys
import os

# Determine project root
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
    
    Args:
        dim: Model dimension
        layer_idx: Layer index for mamba
        drop_path: Drop path rate for stochastic depth
        norm_cls: Normalization layer class
        fused_add_norm: Whether to use fused add norm
        residual_in_fp32: Whether to compute residual in fp32
        expand: Expansion factor for mamba inner dimension
        **mamba_kwargs: Additional arguments for Mamba2
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
        self.dim = dim
        
        # Normalization and regularization
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Mamba2 configuration
        headdim = (dim * expand) // 8
        self.mamba2 = Mamba2(
            d_model=dim,
            headdim=headdim,
            expand=expand,
            use_mem_eff_path=False,
            **mamba_kwargs,
        )
        
        # Output projection for bidirectional processing
        self.out_project = nn.Linear(dim * 2, dim)

    def _reverse_sequences(self, tensor, cu_seqlens):
        """
        Reverse sequences within each batch element based on cu_seqlens.
        
        Args:
            tensor: [1, L, C] input tensor
            cu_seqlens: [B+1] cumulative sequence lengths
            
        Returns:
            reversed_tensor: [1, L, C] tensor with reversed sequences
        """
        reversed_tensor = tensor.clone()
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            reversed_tensor[:, start:end, :] = tensor[:, start:end, :].flip(1)
        return reversed_tensor

    def _process_mamba_forward(self, x_norm, seq_idx, cu_seqlens, inference_params):
        """
        Process forward pass through Mamba2.
        
        Args:
            x_norm: [L, C] normalized input
            seq_idx: sequence indices
            cu_seqlens: cumulative sequence lengths
            inference_params: inference parameters
            
        Returns:
            output: [L, C] processed features
        """
        # Add batch dimension for Mamba2
        u = x_norm.unsqueeze(0)  # [1, L, C]
        
        # Forward pass
        u_out = self.mamba2(
            u,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params
        )
        
        return u_out.squeeze(0)  # [L, C]

    def _process_mamba_bidirectional(self, x_norm, seq_idx, cu_seqlens, inference_params):
        """
        Process bidirectional pass through Mamba2.
        
        Args:
            x_norm: [L, C] normalized input
            seq_idx: sequence indices
            cu_seqlens: cumulative sequence lengths
            inference_params: inference parameters
            
        Returns:
            output: [L, C] processed features from both directions
        """
        # Add batch dimension
        u = x_norm.unsqueeze(0)  # [1, L, C]
        
        # Forward pass
        u_out_fwd = self.mamba2(
            u,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params
        )
        
        # Backward pass with reversed sequences
        u_rev = self._reverse_sequences(u, cu_seqlens)
        u_out_bwd_rev = self.mamba2(
            u_rev,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_params=inference_params
        )
        
        # Reverse the backward output to align with forward
        u_out_bwd = self._reverse_sequences(u_out_bwd_rev, cu_seqlens)
        
        # Concatenate forward and backward outputs
        u_out_combined = torch.cat([u_out_fwd, u_out_bwd], dim=2)  # [1, L, C*2]
        
        # Project back to original dimension
        u_out_projected = self.out_project(u_out_combined)  # [1, L, C]
        
        return u_out_projected.squeeze(0)  # [L, C]

    def forward(
            self, 
            x, 
            pts=None,
            residual=None, 
            inference_params=None,            
            seq_idx=None, 
            cu_seqlens=None,
            bidirectional=False
    ):
        """
        Forward pass through Mamba2Block.
        
        Args:
            x: [L, C] input features
            pts: point counts per scene (unused, kept for compatibility)
            residual: optional residual connection (unused)
            inference_params: parameters for inference
            seq_idx: sequence indices
            cu_seqlens: [B+1] cumulative sequence lengths
            bidirectional: whether to use bidirectional processing
            
        Returns:
            output: [L, C] processed features
            residual: [L, C] residual connection for next layer
        """
        # Pre-normalization
        x_norm = self.norm(x)
        
        # Process through Mamba2
        if bidirectional:
            mamba_out = self._process_mamba_bidirectional(
                x_norm, seq_idx, cu_seqlens, inference_params
            )
        else:
            mamba_out = self._process_mamba_forward(
                x_norm, seq_idx, cu_seqlens, inference_params
            )
        
        # Convert to fp32 if required
        if self.residual_in_fp32:
            mamba_out = mamba_out.to(torch.float32)
        
        # Apply drop path and residual connection
        output = x + self.drop_path(mamba_out)
        
        return output, output
