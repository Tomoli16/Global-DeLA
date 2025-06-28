import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling
from utils.transforms import serialization
from utils.transforms import deserialization
from mamba_layer import Mamba2Block



# Normalerweise speichert PyTorch beim Forward-Pass alle Zwischenergebnisse (Activations) für den Backward-Pass.
# Mit Checkpointing werden diese Zwischenergebnisse nicht gespeichert, sondern die Forward-Pass-Funktion wird bei Bedarf erneut aufgerufen.
def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)


# Andert nichts an der Reihenfolge der Punkte
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
    
class PosEmbedder(nn.Module):
    """
    Positional Embedding via MLP for point coordinates.

    Expects a flat input tensor of shape (P, in_dim), where P = total number of points (batch_size * num_points).
    Outputs an embedding tensor of shape (P, embed_dim).
    """
    def __init__(self,
                 in_dim: int = 3,
                 embed_dim: int = 64,
                 hidden_dim: int = None,
                 act: nn.Module = nn.GELU,
                 bn_momentum: float = 0.1):
        super().__init__()
        # if no hidden dim specified, project directly to embed_dim
        hidden = hidden_dim or embed_dim
        layers = []
        # first linear layer
        layers.append(nn.Linear(in_dim, hidden, bias=False))
        layers.append(nn.BatchNorm1d(hidden, momentum=bn_momentum))
        layers.append(act())
        # second projection to embed_dim
        layers.append(nn.Linear(hidden, embed_dim, bias=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (P, in_dim)
        Returns:
            Tensor of shape (P, embed_dim)
        """
        # Directly apply MLP on flat input
        emb = self.mlp(x)
        return emb


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
        self.mamba2_block = Mamba2Block(dim, depth)
    
    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, knn, pts=None, pts0=None):
        x = x + self.drop_path(self.mlp(x), 0, pts)
        for i in range(self.depth):
            x = x + self.drop_path(self.lfps[i](x, knn), i, pts)
            x_res = x
            x, _ = self.mamba2_block(
                x,
                pts=pts0,
                inference_params=None
            )
            x = x + x_res
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
        dim = args.dims[depth]

        self.grid_size = args.grid_size[depth]  # grid size for serialization
        self.order = args.order
        self.pos_emb = PosEmbedder(in_dim=3, embed_dim=dim, hidden_dim=dim, act=args.act, bn_momentum=args.bn_momentum)


        self.cp = cp = args.use_cp  # Checkpointing
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        
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

        if not last:
            self.sub_stage = Stage(args, depth + 1)

        self.mamba2_block = Mamba2Block(dim, depth)
    
    def local_aggregation(self, x, knn, pts, pts0):
        x = x.unsqueeze(0)  # N x C -> 1 x N x C
        x = self.blk(x, knn, pts, pts0) 
        x = x.squeeze(0) # 1 x N x C -> N x C
        return x

    # Übernimmt das orchestrieren der Mamba2-Block-Operationen
    def mamba2_aggregation(self, x_flat, xyz, pts, inference_params=None):
        """
        x_flat: Tensor [sum_i Ni, C]  (flattened batch of all scenes)
        pts:    Tensor [B]           (#Points per scene)
        """
        # # # 1) Position Embedding
        # xyz = self.pos_emb(xyz)  # xyz: [sum_i Ni, C]
        # x_flat = x_flat + xyz  # add positional embedding to features

        # 2) Serialization

    
        # 2) Mamba2 Block
        u_out, res = self.mamba2_block(
            x_flat,
            pts=pts,
            inference_params=inference_params
        )

        return u_out, res
        

    def forward(self, x, xyz, prev_knn, indices, pts_list):
        """
        x: N x C
        """
        # Durch pop steht hier immer Punktanzahl für das aktuelle Level
        pts0 = pts_list[0]
        
        # downsampling
        if not self.first:
            ids = indices.pop()
            xyz = xyz[ids]
            x = self.skip_proj(x)[ids] + self.lfp(x.unsqueeze(0), prev_knn).squeeze(0)[ids] # LFP + 1x1 Conv

        knn = indices.pop()
        
        # spatial encoding
        N, k = knn.shape    # jeder der N Punkte hat an Stelle knn[i] die Indizes der k nächsten Nachbarn
        nbr = xyz[knn] - xyz.unsqueeze(1)   # xyz[knn] # N x k x 3 xyz.unsqueeze(1) # N x 1 x 3, ergibt relative Koordinaten (dx, dy, dz)
        nbr = torch.cat([nbr, x[knn]], dim=-1).view(-1, 7) if self.first else nbr.view(-1, 3) # # N x k x 7, wenn first, sonst N x k x 3, hängt an (dx, dy, dz) jeweils die Nachbarfeatures an
        if self.training and self.cp:
            nbr.requires_grad_()
        nbr_embed_func = lambda x: self.nbr_embed(x).view(N, k, -1).max(dim=1)[0]   # Pro Punkt, pro Feature, max über die k Nachbarn, N x k x C -> N x C
        nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
        nbr = self.nbr_proj(nbr)
        nbr = self.nbr_bn(nbr)
        x = nbr if self.first else nbr + x  # res connection

        # Local aggregation block
        knn = knn.unsqueeze(0)
        pts = pts_list.pop() if pts_list is not None else None

        x = checkpoint(self.local_aggregation, x, knn, pts, pts0) if self.training and self.cp else self.local_aggregation(x, knn, pts, pts0)

        # Mamba2 aggregation
        # x, _ = self.mamba2_aggregation(x, xyz, pts0)



        


        # get subsequent feature maps (Rekursiver Aufruf)
        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, indices, pts_list)
        else:
            sub_x = sub_c = None

        # regularization (Macht Vorhersagen über relative Positionen)
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

    # xyz: coords x: Feature
    def forward(self, xyz, x, indices, pts_list=None):
        # Flat copy
        indices = indices[:]
        x, closs = self.stage(x, xyz, None, indices, pts_list)
        if self.training:
            return self.head(x), closs
        return self.head(x)

