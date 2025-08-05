import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling
from utils.transforms import serialization
from utils.transforms import deserialization

def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)

class LFP(nn.Module):
    r"""
    Local Feature Propagation Layer
    f = linear(f)
    f_i = bn(max{f_j | j in knn_i} - f_i)
    """
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)
    
    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.proj(x)
        x = knn_edge_maxpooling(x, knn, self.training)
        x = self.bn(x.view(B*N, -1)).view(B, N, -1)
        return x

class ConditionalPE(nn.Module):
    def __init__(self, d_model, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),          # xyz_norm in
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, xyz_norm):
        pe = self.net(xyz_norm) 
        return self.dropout(pe)

class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)
    
    def forward(self, x):
        B, N, C = x.shape
        x = self.mlp(x.view(B*N, -1)).view(B, N, -1)
        return x

class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()

        self.depth = depth
        self.lfps = nn.ModuleList([
            LFP(dim, dim, bn_momentum) for _ in range(depth)
        ])
        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])
        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        #print(drop_rates)
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])
    
    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0, pts)
        for i in range(self.depth):
            x = x + self.drop_path(self.lfps[i](x, knn), i, pts)
            if i % 2 == 1:
                x = x + self.drop_path(self.mlps[i // 2](x), i, pts)
        return x


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth
        self.up_depth = len(args.depths) - 1

        self.first = first = depth == 0
        self.last = last = depth == self.up_depth

        self.k = args.ks[depth]

        self.cp = cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        dim = args.dims[depth]
        nbr_in_dim = 7 if first else 3
        nbr_hid_dim = args.nbr_dims[0] if first else args.nbr_dims[1] // 2
        nbr_out_dim = dim if first else args.nbr_dims[1]
        self.nbr_embed = nn.Sequential(
            nn.Linear(nbr_in_dim, nbr_hid_dim//2, bias=False),
            nn.BatchNorm1d(nbr_hid_dim//2, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim//2, nbr_hid_dim, bias=False),
            nn.BatchNorm1d(nbr_hid_dim, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(nbr_hid_dim, nbr_out_dim, bias=False),
        )
        self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if first else 0.2)
        self.nbr_proj = nn.Identity() if first else nn.Linear(nbr_out_dim, dim, bias=False)

        if not first:
            in_dim = args.dims[depth - 1]
            self.lfp = LFP(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act)
        self.drop = DropPath(args.head_drops[depth])
        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )
        self.order = args.order
        self.grid_size = args.grid_size[depth]  # grid size for serialization

        self.cpe = ConditionalPE(d_model=dim, hidden=dim) 
        self.pe_proj = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )
        self.norm = nn.LayerNorm(dim)


        # Flash Attention Block (separate from traditional Block)
        self.use_flash_attn_block = getattr(args, 'use_flash_attn_blocks', False)
        if self.use_flash_attn_block:
            flash_attn_layers = getattr(args, 'flash_attn_layers', 2)
            self.flash_attn_block = FlashAttentionBlock(
                dim=dim,
                num_layers=flash_attn_layers,
                num_heads=8,
                dropout=0.1,
                mlp_ratio=args.mlp_ratio,
                bn_momentum=args.bn_momentum,
                act=args.act,
                order=self.order,
                grid_size=self.grid_size
            )

        if not last:
            self.sub_stage = Stage(args, depth + 1)
    def build_cu_seqlens(self, seq_lens, device=None, dtype=torch.long):
        """
        Convert per-scene point counts to cumulative sequence lengths tensor.

        Args:
            pts: List[int] or Tensor [B] of point counts per scene
            device: torch.device for output
            dtype: torch.dtype for output
        Returns:
            cu_seqlens: Tensor [B+1]
        """
        # 1) seq_lens in Tensor umwandeln
        if isinstance(seq_lens, list):
            # dtype must be integer
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=device)
        else:
            seq_lens_tensor = seq_lens.to(device=device, dtype=torch.long)

        # 2) cumulative sequence lengths: [0, N0, N0+N1, ...]
        cu_seqlens = torch.cat([
            seq_lens_tensor.new_zeros(1),        # → [0]
            seq_lens_tensor.cumsum(0)            # → [N0, N0+N1, ...]
        ])  # shape = [B+1]
        return cu_seqlens.to(device=device, dtype=dtype)
    
    def normalize_xyz_flat(self, xyz_flat: torch.Tensor,
                        cu_seqlens: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
        """
        xyz_flat   : [total_pts, 3]
        cu_seqlens : [B+1]            – [0, N0, N0+N1, ...]
        Rückgabe: xyz_norm [total_pts, 3] zentriert und auf [-1,1] skaliert pro Szene
        """
        total_pts = xyz_flat.size(0)
        device = xyz_flat.device
        B = cu_seqlens.numel() - 1  # Anzahl Szenen

        # Punktanzahlen pro Szene
        pts = cu_seqlens[1:] - cu_seqlens[:-1]  # [B]

        # Szene-IDs pro Punkt: [0...0,1...1,...]
        scene_ids = torch.arange(B, device=device).repeat_interleave(pts)  # [total_pts]

        # 1) Mittelpunkt pro Szene
        sum_xyz = torch.zeros(B, 3, device=device).scatter_add_(
            0, scene_ids.unsqueeze(-1).expand(-1, 3), xyz_flat
        )  # [B,3]
        count = pts.unsqueeze(-1).to(xyz_flat.dtype)  # [B,1]
        center = sum_xyz / (count + eps)  # [B,3]

        # 2) Zentrieren
        xyz_centered = xyz_flat - center[scene_ids]  # [total_pts,3]

        # 3) Radius pro Punkt
        radii = torch.linalg.norm(xyz_centered, dim=1, keepdim=True)  # [total_pts,1]

        # 4) Max-Radius pro Szene via scatter_reduce_ (amax)
        max_radii = torch.zeros(B, 1, device=device)  # [B,1]
        # scatter_reduce_ schreibt in-place: Index muss die gleiche Shape wie src für dim=0
        # scene_ids.unsqueeze(-1).expand(-1,1) hat Shape [total_pts,1], radii ist [total_pts,1]
        max_radii.scatter_reduce_(0,
                                scene_ids.unsqueeze(-1).expand(-1, 1),
                                radii,
                                reduce='amax',
                                include_self=True)  # [B,1]

        # 5) Normieren (Vermeidung division by zero durch eps)
        xyz_norm = xyz_centered / (max_radii[scene_ids] + eps)  # [total_pts,3]

        return xyz_norm

    def flash_attn_aggregation(self, x_flat, xyz_flat, pts):
        # 1) Choose order
        possible_orders = self.order if isinstance(self.order, list) else [self.order]
        chosen_order = random.choice(possible_orders)

        # 2) Serialization
        xyz_flat, x_flat, _, inverse_order = serialization(
            xyz_flat, x_flat, order=chosen_order, pts=pts, grid_size=self.grid_size
        )

        # 3) Flash Attention
        total, C = x_flat.shape
        device = x_flat.device
        seq_lens = pts if pts is not None else [x_flat.shape[0]]
        cu_seqlens = self.build_cu_seqlens(seq_lens, device=device, dtype=torch.int32)
        max_seqlen = int(max(seq_lens))

        nheads = 8
        head_dim = C // nheads

        to_qkv = torch.nn.Linear(C, 3 * C, bias=False).to(device)
        qkv_flat = to_qkv(x_flat)        # [total, 3*C]
        q, k, v = qkv_flat.chunk(3, dim=-1)  # [total, C], [total, C], [total, C]
        # reshape each to [total, nheads, head_dim]
        q = q.view(total, nheads, head_dim)
        k = k.view(total, nheads, head_dim)
        v = v.view(total, nheads, head_dim)

        # pack into qkv: [total, 3, nheads, head_dim]
        qkv = torch.stack([q, k, v], dim=1)
        x_out = flash_attn_varlen_qkvpacked_func(
            qkv,                  # [total, 3, nheads, head_dim]
            cu_seqlens,           # (B+1,)
            max_seqlen,           # int
            dropout_p=0.1,
            softmax_scale=None,   # standardmäßig 1/sqrt(head_dim)
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
        )  # out: [total, nheads, head_dim]
        x_out_flat = x_out.flatten(1)  # ab Dimension 1 zusammenfalten: [total, C]

        # 4) Deserialization
        xyz_flat, x_out_flat, _, _ = deserialization(
            xyz_ser=xyz_flat,
            feat_ser=x_out_flat,
            x_res_ser=x_out_flat,
            inverse_order=inverse_order,
            layers_outputs_ser=None
        )

        return x_out_flat



    
    def local_aggregation(self, x, knn, pts):
        x = x.unsqueeze(0)
        x = self.blk(x, knn, pts)
        x = x.squeeze(0)
        return x

    def flash_attn_block_aggregation(self, x, xyz, pts):
        """Apply FlashAttentionBlock if enabled"""
        if pts is None:
            pts = [x.shape[0]]
        cu_seqlens = self.build_cu_seqlens(pts, device=x.device, dtype=torch.int32)
        xyz_norm = self.normalize_xyz_flat(xyz, cu_seqlens)   # [total,3]
        pe       = self.cpe(xyz_norm)                         # [total,dim_pe]

        token = torch.cat([x, pe], dim=-1)               # [total,C+pe]
        token = self.pe_proj(token)                           # [total,C]

        # 2) Pre-Norm → Mamba
        token_norm = self.norm(token)

        if hasattr(self, 'flash_attn_block'):
            return self.flash_attn_block(token_norm, xyz, pts)
        else:
            return x

    def forward(self, x, xyz, prev_knn, indices, pts_list):
        """
        x: N x C
        """
        # downsampling
        if not self.first:
            ids = indices.pop()
            xyz = xyz[ids]
            x = self.skip_proj(x)[ids] + self.lfp(x.unsqueeze(0), prev_knn).squeeze(0)[ids]

        knn = indices.pop()
        
        # spatial encoding
        N, k = knn.shape
        nbr = xyz[knn] - xyz.unsqueeze(1)
        nbr = torch.cat([nbr, x[knn]], dim=-1).view(-1, 7) if self.first else nbr.view(-1, 3)
        if self.training and self.cp:
            nbr.requires_grad_()
        nbr_embed_func = lambda x: self.nbr_embed(x).view(N, k, -1).max(dim=1)[0]
        nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr)
        x = nbr if self.first else nbr + x

        # main block
        knn = knn.unsqueeze(0)
        pts = pts_list.pop() if pts_list is not None else None
        x = checkpoint(self.local_aggregation, x, knn, pts) if self.training and self.cp else self.local_aggregation(x, knn, pts)


        # get subsequent feature maps
        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, indices, pts_list)
        else:
            # x = self.flash_attn_aggregation(x, xyz, pts) 
            sub_x = sub_c = None
        
        # regularization
        if self.training:
            rel_k = torch.randint(self.k, (N, 1), device=x.device)
            rel_k = torch.gather(knn.squeeze(0), 1, rel_k).squeeze(1)
            rel_cor = (xyz[rel_k] - xyz)
            rel_cor.mul_(self.cor_std)
            # print(rel_cor.std(dim=0))
            rel_p = x[rel_k] - x
            rel_p = self.cor_head(rel_p)
            closs = F.mse_loss(rel_p, rel_cor)
            sub_c = sub_c + closs if sub_c is not None else closs

        if self.last:
            # Apply FlashAttentionBlock if enabled
            if self.use_flash_attn_block:
                flash_attn_func = lambda x, xyz, pts: self.flash_attn_block_aggregation(x, xyz, pts)
                x = checkpoint(flash_attn_func, x, xyz, pts) if self.training and self.cp else self.flash_attn_block_aggregation(x, xyz, pts)

        # upsampling
        x = self.postproj(x)
        if not self.first:
            back_nn = indices[self.depth-1]
            x = x[back_nn]
        x = self.drop(x)
        sub_x = sub_x + x if sub_x is not None else x

        return sub_x, sub_c

class DelaSemSeg(nn.Module):
    r"""
    DeLA for Semantic Segmentation  

    args:               examples
        depths:         [4, 4, ..., 4]         
        dims:           [128, 256, ..., 512]
        nbr_dims:       [32, 32], dims in spatial encoding || 7->16->32->out->pool | 3->8->16->32->pool->out
        head_dim:       256, hidden dim in cls head
        num_classes:    13
        drop_paths:     [0., 0., ..., 0.1], in-stage drop path rate, can be list of lists, len(dp[i]) = depth[i]
        head_drops:     [0., 0.05, ..., 0.2], scale wise drop rate before cls head
        bn_momentum:    0.02         
        act:            nn.GELU
        mlp_ratio:      2, can be float
        use_cp:         False, enable gradient checkpoint to save memory
                        If True, blocks and spatial encoding are checkpointed
    """
    def __init__(self, args):
        super().__init__()

        # bn momentum for checkpointed layers
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5

        self.stage = Stage(args)

        hid_dim = args.head_dim
        out_dim = args.num_classes

        self.head = nn.Sequential(
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(hid_dim, out_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None):
        indices = indices[:]
        x, closs = self.stage(x, xyz, None, indices, pts_list)
        if self.training:
            return self.head(x), closs
        return self.head(x)

class FlashAttentionBlock(nn.Module):
    """
    Flash Attention Block with multiple attention layers and skip connections
    """
    def __init__(self, dim, num_layers=2, num_heads=8, dropout=0.1, mlp_ratio=4, 
                 bn_momentum=0.02, act=nn.GELU, order=None, grid_size=0.04):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.order = order or ["xyz", "hilbert", "z"]
        self.grid_size = grid_size
        
        # Multiple attention layers
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'qkv_proj': nn.Linear(dim, 3 * dim, bias=False),
                'out_proj': nn.Linear(dim, dim, bias=False),
                'norm': nn.LayerNorm(dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        
        # MLP layers for each attention layer
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, int(dim * mlp_ratio)),
                act(),
                nn.Dropout(dropout),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Skip connection projections
        self.skip_projections = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) if i > 0 else nn.Identity()
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)
        
    def build_cu_seqlens(self, seq_lens, device, dtype=torch.int32):
        """Build cumulative sequence lengths for flash attention"""
        if not isinstance(seq_lens, torch.Tensor):
            seq_lens_tensor = torch.tensor(seq_lens, dtype=dtype, device=device)
        else:
            seq_lens_tensor = seq_lens.to(device=device, dtype=dtype)

        cu_seqlens = torch.cat([
            seq_lens_tensor.new_zeros(1),
            seq_lens_tensor.cumsum(0)
        ])
        return cu_seqlens.to(device=device, dtype=dtype)

    def apply_flash_attention(self, x_flat, xyz_flat, pts, layer_idx):
        """Apply flash attention to flattened features"""
        # Choose serialization order
        possible_orders = self.order if isinstance(self.order, list) else [self.order]
        chosen_order = random.choice(possible_orders)

        # Serialization
        xyz_flat, x_flat, _, inverse_order = serialization(
            xyz_flat, x_flat, order=chosen_order, pts=pts, grid_size=self.grid_size
        )

        # Prepare for flash attention
        total, C = x_flat.shape
        device = x_flat.device
        seq_lens = pts if pts is not None else [x_flat.shape[0]]
        cu_seqlens = self.build_cu_seqlens(seq_lens, device=device, dtype=torch.int32)
        max_seqlen = int(max(seq_lens))

        # Get QKV projections
        qkv_flat = self.attention_layers[layer_idx]['qkv_proj'](x_flat)
        q, k, v = qkv_flat.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)

        # Pack into qkv format
        qkv = torch.stack([q, k, v], dim=1)
        
        # Apply flash attention
        x_out = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
        )
        
        # Project output
        x_out_flat = x_out.flatten(1)
        x_out_flat = self.attention_layers[layer_idx]['out_proj'](x_out_flat)

        # Deserialization
        xyz_flat, x_out_flat, _, _ = deserialization(
            xyz_ser=xyz_flat,
            feat_ser=x_out_flat,
            x_res_ser=x_out_flat,
            inverse_order=inverse_order,
            layers_outputs_ser=None
        )

        return x_out_flat

    def forward(self, x, xyz, pts=None):
        """
        Forward pass with multiple flash attention layers and skip connections
        
        Args:
            x: Input features [N, C]
            xyz: Point coordinates [N, 3]
            pts: Points per sequence (for batched processing)
        """
        residual_connections = []
        
        for i in range(self.num_layers):
            # Store residual connection
            if i == 0:
                residual = x
            else:
                # Apply skip projection for deeper layers
                residual = self.skip_projections[i](x)
            
            residual_connections.append(residual)
            
            # Layer normalization
            x_norm = self.attention_layers[i]['norm'](x)
            
            # Flash attention
            x_attn = self.apply_flash_attention(x_norm, xyz, pts, i)
            x_attn = self.attention_layers[i]['dropout'](x_attn)
            
            # First residual connection (attention)
            x = residual + x_attn
            
            # MLP with second residual connection
            x_mlp = self.mlp_layers[i](x)
            x = x + x_mlp
            
            # Additional skip connections from previous layers
            if i > 0:
                # Add skip connections from all previous layers with learnable weights
                skip_weight = 0.1 / i  # Decrease weight for older connections
                for j in range(i):
                    x = x + skip_weight * residual_connections[j]
        
        # Final layer normalization
        x = self.final_norm(x)
        
        return x

