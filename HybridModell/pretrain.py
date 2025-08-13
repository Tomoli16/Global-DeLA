import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
# ChamferDistance with fallback to MSE
try:
    from chamferdist import ChamferDistance
    CHAMFER_AVAILABLE = True
except ImportError:
    CHAMFER_AVAILABLE = False
    print("Warning: chamferdist not available, using MSE loss instead")
import wandb
import argparse
from pathlib import Path
import sys
import os
from datetime import datetime
import math

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).absolute().parent.parent))

from pretrainer import MaskedPreTrainer
# Import will be done dynamically based on dataset choice
# from scannetv2 import ScanNetV2, scan_collate_fn
# from s3dis import S3DIS, s3dis_collate_fn
from config import scan_args, dela_args, batch_size

def create_output_dirs(run_id):
    """Create necessary output directories"""
    output_dir = Path("output/model") / f"pretrain_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def apply_cd_loss_padding(recon_pos, gt_pos, target_points=16*18000):
    """
    Apply padding specifically for CD loss computation to achieve target dimensions.
    
    Args:
        recon_pos: Reconstructed positions tensor [N_masked, C]
        gt_pos: Ground truth positions tensor [N_masked, C]
        target_points: Target total number of points (B * N)
    
    Returns:
        padded_recon: Padded reconstructed positions
        padded_gt: Padded ground truth positions
    """
    actual_points = recon_pos.shape[0]
    C = recon_pos.shape[1]
    
    if actual_points < target_points:
        # Calculate padding needed
        padding_needed = target_points - actual_points
        
        # Create padding by repeating the last point
        if actual_points > 0:
            last_recon = recon_pos[-1:].expand(padding_needed, -1)
            last_gt = gt_pos[-1:].expand(padding_needed, -1)
        else:
            # Fallback: use zeros if no points available
            last_recon = torch.zeros(padding_needed, C, device=recon_pos.device, dtype=recon_pos.dtype)
            last_gt = torch.zeros(padding_needed, C, device=gt_pos.device, dtype=gt_pos.dtype)
        
        # Concatenate original points with padding
        padded_recon = torch.cat([recon_pos, last_recon], dim=0)
        padded_gt = torch.cat([gt_pos, last_gt], dim=0)
    elif actual_points > target_points:
        # Truncate if we have too many points
        padded_recon = recon_pos[:target_points]
        padded_gt = gt_pos[:target_points]
    else:
        # Exact match - no padding needed
        padded_recon = recon_pos
        padded_gt = gt_pos
    
    return padded_recon, padded_gt

def pretrain_model():
    # Argument parser
    parser = argparse.ArgumentParser(description='Pretrain with MaskedPreTrainer')
    parser.add_argument('--model', type=str, default='dela_semseg', 
                       choices=['dela_semseg', 'dela_semseg_attn'],
                       help='Model type to use for pretraining')
    parser.add_argument('--dataset', type=str, default='scannetv2',
                       choices=['scannetv2', 's3dis'],
                       help='Dataset to use for pretraining')
    parser.add_argument('--run_id', type=str, default='pretrain_002',
                       help='Unique identifier for this pretraining run')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of pretraining epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.6,
                       help='Ratio of points to mask (0.0-1.0)')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                       help='Log frequency in batches')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='Use Weights & Biases for logging')
    parser.add_argument('--lam_closs', type=float, default=0.1, help='Weight for consistency loss regularization (closs)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for LR schedule before cosine decay')
    args = parser.parse_args()

    print(f"Starting pretraining with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Run ID: {args.run_id}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {batch_size}")  # Fixed: use imported batch_size
    print(f"  Learning Rate: {args.lr}")
    print(f"  Mask Ratio: {args.mask_ratio}")

    # Create output directories
    output_dir = create_output_dirs(args.run_id)
    
    # Import model dynamically
    if args.model == "dela_semseg":
        from hybridmodel import DelaSemSeg as ModelClass
    elif args.model == "dela_semseg_attn":
        from delasemseg_attn import DelaSemSeg as ModelClass
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Import dataset dynamically based on choice
    if args.dataset == "scannetv2":
        from scannetv2 import ScanNetV2 as DatasetClass, scan_collate_fn as collate_fn
        dataset_args = scan_args
        input_dim = 10  # ScanNetV2: 3 RGB + 1 height + 3 normals + 3 xyz
    elif args.dataset == "s3dis":
        from s3dis import S3DIS as DatasetClass, s3dis_collate_fn as collate_fn
        from config_s3dis import s3dis_args
        dataset_args = s3dis_args
        input_dim = 7   # S3DIS: 3 RGB + 1 height + 3 xyz (no normals)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Initialize model and pretrainer
    print("Initializing model...")
    
    # Set input feature dimension for the model
    if hasattr(dela_args, 'input_feature_dim'):
        dela_args.input_feature_dim = input_dim
    else:
        setattr(dela_args, 'input_feature_dim', input_dim)
    
    base_model = ModelClass(dela_args).cuda()
    
    pretrainer = MaskedPreTrainer(
        encoder=base_model.backbone,  # Use only the encoder part
        mask_ratio=args.mask_ratio,
        embed_dim=dela_args.embed_dim,
        input_dim=input_dim
    ).cuda()
    
    print(f"Model initialized with {sum(p.numel() for p in pretrainer.parameters())} parameters")
    
    # Data loader for pretraining (without labels)
    print("Setting up data loader...")
    
    # Create dataset based on choice
    if args.dataset == "scannetv2":
        pretrain_dataset = DatasetClass(dataset_args, partition="train", loop=2, train=True)
    else:  # s3dis
        pretrain_dataset = DatasetClass(dataset_args, partition="!5", loop=2, train=True)  # S3DIS uses different partition format
    
    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        persistent_workers=True
    )
    
    print(f"Dataset loaded: {len(pretrain_dataset)} samples, {len(pretrain_loader)} batches per epoch")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(pretrain_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        # Cosine decay from base lr to base_lr * 0.01
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine * 0.99 + 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler()

    # Use ChamferDistance or MSE loss
    if CHAMFER_AVAILABLE:
        chamfer_loss = ChamferDistance()
        print("Using ChamferDistance loss")
    else:
        print("Using MSE loss as fallback")

    # Wandb logging
    if args.use_wandb:
        wandb.init(
            project="DeLA-Pretraining",
            name=f"pretrain_{args.run_id}",
            config=vars(args)
        )
    
    # Training loop
    print("Starting pretraining...")
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        pretrainer.train()
        epoch_loss = 0
        num_batches = 0
        num_valid_batches = 0
        
        for batch_idx, (xyz, feature, indices, pts, _) in enumerate(pretrain_loader):
            # Move to GPU
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            pts = pts.tolist()[::-1]  # Convert to list and reverse, like in S3DIS
            
            optimizer.zero_grad()
            
            try:
                with autocast():
                    recon_pos, gt_pos, closs = pretrainer(xyz, feature, indices, pts)
                    
                    # Validierung der Outputs
                    if recon_pos is None or gt_pos is None:
                        print(f"Batch {batch_idx}: Pretrainer returned None, skipping...")
                        continue
                    
                    if recon_pos.numel() == 0 or gt_pos.numel() == 0:
                        print(f"Batch {batch_idx}: Empty reconstruction tensors, skipping...")
                        continue
                    
                    if CHAMFER_AVAILABLE:
                        # Ensure (B, N, 3)
                        recon_pos = recon_pos.float().contiguous()
                        gt_pos = gt_pos.float().contiguous()
                        if recon_pos.dim() == 2:  # (N,3) -> (1,N,3)
                            recon_pos = recon_pos.unsqueeze(0)
                            gt_pos = gt_pos.unsqueeze(0)
                        loss_rec = chamfer_loss(recon_pos, gt_pos, bidirectional=True, point_reduction='mean')
                    else:
                        loss_rec = F.mse_loss(recon_pos, gt_pos)

                    closs_val = closs if closs is not None else torch.tensor(0.0, device=loss_rec.device)
                    loss = loss_rec + args.lam_closs * closs_val
                    
                    # Validierung des Loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Batch {batch_idx}: Invalid loss {loss.item()}, skipping...")
                        continue
                        
                    num_valid_batches += 1
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Logging
                if batch_idx % args.log_freq == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    masked_points = recon_pos.shape[1] if recon_pos.dim() == 3 else recon_pos.shape[0]
                    print(f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(pretrain_loader)} | "
                          f"Loss: {loss.item():.6f} | Rec: {loss_rec.item():.6f} | cLoss: {closs_val.item():.6f} | LR: {current_lr:.2e} | "
                          f"Masked Points: {masked_points}")

                    if args.use_wandb:
                        wandb.log({
                            "batch_loss": loss.item(),
                            "batch_loss_rec": loss_rec.item(),
                            "batch_closs": closs_val.item(),
                            "learning_rate": current_lr,
                            "masked_points": masked_points,
                            "global_step": global_step
                        })
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"  xyz shape: {xyz.shape if hasattr(xyz, 'shape') else type(xyz)}")
                print(f"  feature shape: {feature.shape if hasattr(feature, 'shape') else type(feature)}")
                print(f"  pts: {pts} (type: {type(pts)})")
                continue
        
        # Epoch statistics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch:3d} Summary:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Valid Batches: {num_valid_batches}/{len(pretrain_loader)}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_loss": avg_loss,
                "valid_batches": num_valid_batches,
                "epoch_lr": optimizer.param_groups[0]["lr"]
            })
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': pretrainer.state_dict(),
                'encoder_state_dict': pretrainer.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / "best_checkpoint.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': pretrainer.state_dict(),
                    'encoder_state_dict': pretrainer.encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'args': vars(args)
                }, best_path)
                print(f"  Best model saved: {best_path}")
        
        print("-" * 60)
    
    # Save final pretrained encoder
    final_encoder_path = output_dir / "pretrained_encoder.pt"
    torch.save(pretrainer.encoder.state_dict(), final_encoder_path)
    print(f"\nFinal pretrained encoder saved: {final_encoder_path}")
    
    # Save training summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Pretraining Summary\n")
        f.write(f"==================\n")
        f.write(f"Run ID: {args.run_id}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Final Loss: {avg_loss:.6f}\n")
        f.write(f"Best Loss: {best_loss:.6f}\n")
        f.write(f"Mask Ratio: {args.mask_ratio}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Batch Size: {batch_size}\n")  # Fixed: use imported batch_size
        f.write(f"Completed: {datetime.now()}\n")
    
    if args.use_wandb:
        wandb.finish()
    
    print(f"\nPretraining completed! Best loss: {best_loss:.6f}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    pretrain_model()
