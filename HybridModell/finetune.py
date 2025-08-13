import random, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from time import time, sleep
from config_s3dis import (
    s3dis_args, s3dis_warmup_args, dela_args, batch_size, learning_rate, 
    epoch, warmup, label_smoothing, model_type, run_id, processed_data_path,
    configure_flash_attention, configure_mamba2, configure_run
)
from hybridmodel import DelaSemSeg as HybridModel
import wandb

# Parse command line arguments
parser = argparse.ArgumentParser(description='Finetune DeLA on S3DIS with pretrained encoder weights')
parser.add_argument('--pretrained_path', type=str, required=True,
                    help='Path to pretrained encoder weights (.pt file)')
parser.add_argument('--run_id', type=str, default=run_id,
                    help='Run ID for logging and model saving')
parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='Gradient clipping value (default: 1.0)')
parser.add_argument('--partition', type=str, default="!5", 
                    help='S3DIS partition for training (default: "!5" - all areas except 5)')
parser.add_argument('--val_partition', type=str, default="5",
                    help='S3DIS partition for validation (default: "5" - area 5)')
parser.add_argument('--freeze_encoder', action='store_true',
                    help='Freeze encoder weights during finetuning (only train head)')
parser.add_argument('--lr_scale_encoder', type=float, default=0.1,
                    help='Learning rate scale for encoder when not frozen (default: 0.1)')
args = parser.parse_args()

# Use HybridModel from hybridmodel.py
ModelClass = HybridModel

torch.set_float32_matmul_precision("high")

def load_pretrained_encoder(model, pretrained_path):
    """
    Lädt vortrainierte Encoder-Gewichte und ignoriert Decoder-Parameter
    """
    print(f"Loading pretrained weights from: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")
    
    # Lade pretrained state dict
    pretrained_state = torch.load(pretrained_path, map_location='cpu')
    pretrained_state = pretrained_state["model_state_dict"]

    # Falls es ein util.save_state Format ist, extrahiere das model state dict
    if 'model' in pretrained_state:
        pretrained_state = pretrained_state['model']
    
    # Aktuelles Model state dict
    model_state = model.state_dict()
    
    # Filtere Decoder-Parameter aus (alles was mit 'decoder' beginnt)
    encoder_state = {}
    loaded_keys = []
    skipped_keys = []
    
    for key, value in pretrained_state.items():
        # Überspringe Decoder-Parameter
        if any(skip_prefix in key.lower() for skip_prefix in ['decoder', 'head']):
            skipped_keys.append(key)
            continue
            
        # Überspringe Parameter die nicht im aktuellen Model existieren
        if key not in model_state:
            skipped_keys.append(key)
            continue
            
        # Überspringe Parameter mit unterschiedlichen Shapes
        if value.shape != model_state[key].shape:
            skipped_keys.append(key)
            print(f"Shape mismatch for {key}: pretrained {value.shape} vs model {model_state[key].shape}")
            continue
            
        encoder_state[key] = value
        loaded_keys.append(key)
    
    # Lade die gefilterten Gewichte
    model.load_state_dict(encoder_state, strict=False)
    
    print(f"Successfully loaded {len(loaded_keys)} encoder parameters")
    print(f"Skipped {len(skipped_keys)} parameters (decoder or incompatible)")
    
    return model

def setup_optimizer_with_different_lr(model, base_lr, encoder_lr_scale, freeze_encoder=False):
    """
    Setup optimizer with different learning rates for encoder and decoder
    """
    if freeze_encoder:
        # Nur Decoder-Parameter trainieren
        decoder_params = []
        for name, param in model.named_parameters():
            if any(decoder_prefix in name.lower() for decoder_prefix in ['decoder']):
                decoder_params.append(param)
            else:
                param.requires_grad = False
        
        optimizer = create_optimizer_v2(decoder_params, lr=base_lr, weight_decay=5e-2)
        print(f"Frozen encoder mode: Only training decoder with lr={base_lr}")
        
    else:
        # Verschiedene Learning Rates für Encoder und Decoder
        encoder_params = []
        decoder_params = []
        
        for name, param in model.named_parameters():
            if any(decoder_prefix in name.lower() for decoder_prefix in ['decoder']):
                decoder_params.append(param)
            else:
                encoder_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': base_lr * encoder_lr_scale},
            {'params': decoder_params, 'lr': base_lr}
        ]
        
        optimizer = create_optimizer_v2(param_groups, lr=base_lr, weight_decay=5e-2)
        print(f"Different LR mode: Encoder lr={base_lr * encoder_lr_scale}, Decoder lr={base_lr}")
    
    return optimizer

def warmup_fn(model, dataset, grad_clip=1.0):
    model.train()
    traindlr = DataLoader(dataset, batch_size=len(dataset), collate_fn=s3dis_collate_fn, pin_memory=True, num_workers=6)
    for xyz, feature, indices, pts, y in traindlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            p, closs = model(xyz, feature, indices, pts)
            # S3DIS hat keine ignore_index wie ScanNet (kein unlabeled class)
            loss = F.cross_entropy(p, y, label_smoothing=label_smoothing) + closs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# Setup directories
cur_id = args.run_id
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
errfile = f"output/log/{cur_id}/err.log"
logfile = open(logfile, "a", 1)
errfile = open(errfile, "a", 1)
sys.stdout = logfile
sys.stderr = errfile

print("="*80)
print("S3DIS FINETUNING CONFIGURATION")
print("="*80)
print(f"Model: HybridModel (DelaSemSeg)")
print(f"Pretrained weights: {args.pretrained_path}")
print(f"Training partition: {args.partition}")
print(f"Validation partition: {args.val_partition}")
print(f"Freeze encoder: {args.freeze_encoder}")
if not args.freeze_encoder:
    print(f"Encoder LR scale: {args.lr_scale_encoder}")
print(f"Flash Attention: {dela_args.use_flash_attn_blocks}")
if dela_args.use_flash_attn_blocks:
    print(f"Flash Attention Layers: {dela_args.flash_attn_layers}")
print(f"Mamba2: {dela_args.run_mamba}")
if dela_args.run_mamba:
    print(f"Mamba2 Depth: {dela_args.mamba_depth}")
print(f"Gradient Clipping: {args.grad_clip}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print("="*80)

# Initialize wandb
wandb.init(
    project="DeLA-S3DIS-Finetune",
    name=f"run_{cur_id}_hybrid",
    config={
        "model_type": "hybrid",
        "pretrained_path": args.pretrained_path,
        "freeze_encoder": args.freeze_encoder,
        "lr_scale_encoder": args.lr_scale_encoder,
        "training_partition": args.partition,
        "validation_partition": args.val_partition,
        "use_flash_attn_blocks": dela_args.use_flash_attn_blocks,
        "flash_attn_layers": dela_args.flash_attn_layers,
        "run_mamba": dela_args.run_mamba,
        "mamba_depth": dela_args.mamba_depth,
        "mamba_drop_path_rate": dela_args.mamba_drop_path_rate,
        "grad_clip": args.grad_clip,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epoch,
        "warmup_epochs": warmup,
        "label_smoothing": label_smoothing,
        **vars(s3dis_args),
        **vars(dela_args),
    }
)

# Setup data loaders
print("Setting up data loaders...")
traindlr = DataLoader(S3DIS(s3dis_args, partition=args.partition, loop=30, train=True), 
                      batch_size=batch_size, collate_fn=s3dis_collate_fn, shuffle=True, 
                      pin_memory=True, persistent_workers=True, drop_last=True, num_workers=16)

testdlr = DataLoader(S3DIS(s3dis_args, partition=args.val_partition, loop=1, train=False), 
                     batch_size=1, collate_fn=s3dis_collate_fn, pin_memory=True, 
                     persistent_workers=True, num_workers=16)

step_per_epoch = len(traindlr)

# Initialize model
print("Initializing model...")
model = ModelClass(dela_args).cuda()

# Load pretrained encoder weights
model = load_pretrained_encoder(model, args.pretrained_path)

# Setup optimizer with potentially different learning rates
optimizer = setup_optimizer_with_different_lr(
    model, learning_rate, args.lr_scale_encoder, args.freeze_encoder
)

scheduler = CosineLRScheduler(optimizer, t_initial=epoch * step_per_epoch, lr_min=learning_rate/10000,
                             warmup_t=warmup*step_per_epoch, warmup_lr_init=learning_rate/20)
scaler = GradScaler()

# Resume training if checkpoint exists
resume = False
start_epoch = 0
if resume:
    checkpoint_path = f"output/model/{cur_id}/last.pt"
    if os.path.exists(checkpoint_path):
        start_epoch = util.load_state(checkpoint_path, model=model, optimizer=optimizer, scaler=scaler)["start_epoch"]
        print(f"Resumed from epoch {start_epoch}")

scheduler_step = start_epoch * step_per_epoch

# Initialize metrics
metric = util.Metric(13)  # S3DIS hat 13 Klassen
ttls = util.AverageMeter() 
corls = util.AverageMeter() 
grad_norms = util.AverageMeter()
best = 0

# Warmup
print("Starting warmup...")
warmup_fn(model, S3DIS(s3dis_warmup_args, partition=args.partition, loop=batch_size, warmup=True), args.grad_clip)

print("Starting training...")
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    metric.reset()
    corls.reset()
    grad_norms.reset()
    now = time()
    
    for xyz, feature, indices, pts, y in traindlr:
        lam = scheduler_step/(epoch*step_per_epoch)
        lam = 3e-3 ** lam * 0.2
        scheduler.step(scheduler_step)
        scheduler_step += 1
        
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        
        with autocast():
            p, closs = model(xyz, feature, indices, pts)
            # S3DIS hat keine ignore_index (alle Punkte sind gelabelt)
            loss = F.cross_entropy(p, y, label_smoothing=label_smoothing)
        
        metric.update(p.detach(), y)
        ttls.update(loss.item())
        corls.update(closs.item())
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss + lam*closs).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_norms.update(grad_norm.item())
        
        scaler.step(optimizer)
        scaler.update()
    
    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:")
    train_miou = metric.miou
    train_loss = ttls.avg
    train_closs = corls.avg
    train_grad_norm = grad_norms.avg
    
    # Log training metrics
    wandb.log({
        "epoch": i,
        "train_loss": train_loss,
        "train_closs": train_closs,
        "train_miou": train_miou,
        "train_macc": metric.macc,
        "train_acc": metric.acc,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "grad_norm": train_grad_norm,
    }, step=i)
    
    # Validation
    model.eval()
    metric.reset()
    val_ttls = util.AverageMeter()
    with torch.no_grad():
        for xyz, feature, indices, pts, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            y = y.cuda(non_blocking=True)
            pts = pts.tolist()[::-1]
            
            with autocast():
                p = model(xyz, feature, indices, pts)
                loss = F.cross_entropy(p, y, label_smoothing=label_smoothing)
            
            metric.update(p, y)
            val_ttls.update(loss.item())
    
    metric.print("val:  ")
    val_miou = metric.miou
    val_macc = metric.macc
    val_acc = metric.acc
    val_loss = val_ttls.avg
    duration = time() - now
    
    # Log validation metrics
    wandb.log({
        "epoch": i,
        "val_loss": val_loss,
        "val_miou": val_miou,
        "val_macc": val_macc,
        "val_acc": val_acc,
        "duration": duration,
    }, step=i)
    
    print(f"val loss: {round(val_loss, 4)}")
    print(f"duration: {duration}")
    
    cur = metric.miou
    if best < cur:
        best = cur
        print("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)
    
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler, start_epoch=i+1)

print(f"Training completed! Best mIoU: {best:.4f}")
wandb.finish()
