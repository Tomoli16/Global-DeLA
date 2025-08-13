import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import random
import sys

class MaskedPreTrainer(nn.Module):
    def __init__(self, encoder, mask_ratio=0.6, embed_dim=384, input_dim=None):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # mask_token sollte die gleiche Dimension haben wie die Input-Features
        # Wenn input_dim nicht gegeben ist, verwende embed_dim
        self.input_dim = input_dim if input_dim is not None else embed_dim
        self.mask_token = nn.Parameter(torch.randn(1, self.input_dim))
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)   # Rekonstruiere GT Positionen (XYZ)
        )
        # Initialize mask token
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, xyz, x, indices, pts_list):
        device = x.device
        
        # Dynamische Anpassung der mask_token Dimension
        input_feature_dim = x.shape[-1]
        if self.mask_token.shape[-1] != input_feature_dim:
            # Erstelle neuen mask_token mit der richtigen Dimension
            mask_token_adapted = nn.Parameter(torch.randn(1, input_feature_dim, device=device))
            trunc_normal_(mask_token_adapted, std=0.02)
        else:
            mask_token_adapted = self.mask_token
        
        # pts_list ist eine Liste von Listen: [[pts_level0_scene0, pts_level0_scene1, ...], [pts_level1_scene0, ...], ...]
        # Wir brauchen die Punktanzahl für das erste (höchste) Level
        if isinstance(pts_list, list) and len(pts_list) > 0:
            # Erste Level (level 0) verwenden - das ist die original Punktanzahl
            pts_current_level = pts_list[-1] if isinstance(pts_list[0], list) else pts_list
        else:
            raise ValueError(f"Invalid pts_list structure: {pts_list}")
        
        # Finde die minimale Anzahl an Punkten im aktuellen Level
        min_points = min(int(pts) for pts in pts_current_level if int(pts) > 0)
        num_masked = max(1, int(min_points * self.mask_ratio))  # Gleiche Anzahl für alle Szenen
        
        
        # Effiziente GPU-basierte Masken-Generierung
        mask_idx_list = []
        gt_pos_list = []
        masked_feat_list = []
        
        # Offset für das Durchlaufen der flachen Tensoren
        offset = 0
        for i in range(len(pts_current_level)):
            num_points = int(pts_current_level[i])
            
            # Validierung der Punkt-Anzahl
            if num_points <= 0:
                print(f"Warning: Scene {i} has {num_points} points, skipping...")
                continue
            
            # GPU-effiziente Masken-Erstellung mit fester Anzahl num_masked
            try:
                # Für Szenen mit weniger Punkten als num_masked, nutze alle verfügbaren Punkte
                actual_num_masked = min(num_masked, num_points)
                rand_indices = torch.randperm(num_points, device=device)[:actual_num_masked]
                mask = torch.zeros(num_points, dtype=torch.bool, device=device)
                mask[rand_indices] = True
            except Exception as e:
                print(f"Error creating mask for scene {i}: {e}")
                offset += num_points
                continue
            
            # Extrahiere Szenen-spezifische Daten aus flachen Tensoren
            try:
                scene_xyz = xyz[offset:offset + num_points]
                scene_x = x[offset:offset + num_points]
                
                # Validierung der Dimensionen
                if scene_xyz.shape[0] != num_points or scene_x.shape[0] != num_points:
                    print(f"Dimension mismatch in scene {i}: expected {num_points}, got xyz:{scene_xyz.shape[0]}, x:{scene_x.shape[0]}")
                    offset += num_points
                    continue
                    
            except Exception as e:
                print(f"Error slicing tensors for scene {i}: {e}")
                offset += num_points
                continue
            
            mask_idx_list.append(mask)
            gt_pos_list.append(scene_xyz[mask])
            
            # Für Encoder: Verwende mask_token für maskierte Positionen
            try:
                masked_x = scene_x.clone()
                # Verwende den adaptierten mask_token mit der richtigen Dimension
                actual_num_masked = mask.sum().item()
                masked_x[mask] = mask_token_adapted.expand(actual_num_masked, -1)
                masked_feat_list.append(masked_x)
                    
            except Exception as e:
                print(f"Error masking features for scene {i}: {e}")
                print(f"  scene_x shape: {scene_x.shape}")
                print(f"  mask_token shape: {mask_token_adapted.shape}")
                print(f"  mask sum: {mask.sum().item()}")
                offset += num_points
                continue
            
            offset += num_points
        
        # Validierung vor Konkatenierung
        if not masked_feat_list:
            print("Warning: No valid scenes found, returning empty tensors")
            return torch.empty((0, 3), device=device), torch.empty((0, 3), device=device)
        
        try:
            # Concatenate masked features für Encoder
            masked_feats = torch.cat(masked_feat_list, dim=0)
            
            # Encodiere mit maskierten Features (xyz bleibt wie es ist)
            encoded_feats, closs = self.encoder(xyz, masked_feats, indices, pts_list)
        except Exception as e:
            print(f"Error in encoder: {e}")
            return (torch.empty((0, 3), device=device),
                    torch.empty((0, 3), device=device),
                    torch.tensor(0.0, device=device))

        # Dekodiere nur maskierte Positionen
        all_masked_feats = []
        all_gt_pos = []
        offset = 0
        for i, mask in enumerate(mask_idx_list):
            if i >= len(pts_current_level):
                break
            num_points = int(pts_current_level[i])
            
            try:
                scene_feats = encoded_feats[offset:offset + num_points]
                masked_scene_feats = scene_feats[mask]
                
                # Padding zu num_masked falls weniger maskierte Punkte vorhanden
                if masked_scene_feats.shape[0] < num_masked:
                    padding_size = num_masked - masked_scene_feats.shape[0]
                    padding = torch.zeros(padding_size, masked_scene_feats.shape[1], device=device)
                    masked_scene_feats = torch.cat([masked_scene_feats, padding], dim=0)
                elif masked_scene_feats.shape[0] > num_masked:
                    # Truncate falls mehr als num_masked (sollte nicht passieren)
                    masked_scene_feats = masked_scene_feats[:num_masked]
                
                all_masked_feats.append(masked_scene_feats)
                
                # Gleiche Behandlung für GT Positionen
                scene_gt_pos = gt_pos_list[i]
                if scene_gt_pos.shape[0] < num_masked:
                    padding_size = num_masked - scene_gt_pos.shape[0]
                    padding = torch.zeros(padding_size, 3, device=device)
                    scene_gt_pos = torch.cat([scene_gt_pos, padding], dim=0)
                elif scene_gt_pos.shape[0] > num_masked:
                    scene_gt_pos = scene_gt_pos[:num_masked]
                
                all_gt_pos.append(scene_gt_pos)
                
            except Exception as e:
                print(f"Error decoding scene {i}: {e}")
            
            offset += num_points
        
        # Batch-Dekodierung mit einheitlicher Form (B, num_masked, C)
        if all_masked_feats:
            try:
                # Stack zu (B, num_masked, feature_dim)
                masked_encoded = torch.stack(all_masked_feats, dim=0)  # (B, num_masked, feature_dim)
                gt_pos = torch.stack(all_gt_pos, dim=0)  # (B, num_masked, 3)

                # Reshape für Decoder: (B*num_masked, feature_dim)
                B, num_masked_actual, feature_dim = masked_encoded.shape
                masked_encoded_flat = masked_encoded.view(-1, feature_dim)
                recon_pos_flat = self.decoder(masked_encoded_flat)  # (B*num_masked, 3)

                # Reshape zurück zu (B, num_masked, 3)
                recon_pos = recon_pos_flat.view(B, num_masked_actual, 3)

                return recon_pos, gt_pos, closs if closs is not None else torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"Error in final decoding: {e}")
                return (torch.empty((0, num_masked, 3), device=device),
                        torch.empty((0, num_masked, 3), device=device),
                        torch.tensor(0.0, device=device))
        # Fallback wenn keine maskierten Feats
        return (torch.empty((0, num_masked, 3), device=device),
                torch.empty((0, num_masked, 3), device=device),
                torch.tensor(0.0, device=device))
