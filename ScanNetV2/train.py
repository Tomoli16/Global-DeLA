import random, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from scannetv2 import ScanNetV2, scan_collate_fn
from torch.utils.data import DataLoader
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from time import time, sleep
from config import scan_args, scan_warmup_args, dela_args, batch_size, learning_rate as lr, epoch, warmup, label_smoothing as ls, model_type, run_id, configure_flash_attention, configure_mamba2, configure_run
import wandb

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train DeLA on ScanNetV2')
parser.add_argument('--model', type=str, default=model_type, choices=['dela_semseg', 'dela_semseg_attn'],
                    help='Model type to use (default: from config.py)')
parser.add_argument('--run_id', type=str, default=run_id,
                    help='Run ID for logging and model saving')
parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='Gradient clipping value (default: 1.0)')
args = parser.parse_args()

# Override config with command line arguments
model_type = args.model

# Dynamic model import based on config
if model_type == "dela_semseg":
    from delasemseg import DelaSemSeg as ModelClass
elif model_type == "dela_semseg_attn":
    from delasemseg_attn import DelaSemSeg as ModelClass
else:
    raise ValueError(f"Unknown model_type: {model_type}. Use 'dela_semseg' or 'dela_semseg_attn'")

torch.set_float32_matmul_precision("high")

def warmup_fn(model, dataset, grad_clip=1.0):
    model.train()
    traindlr = DataLoader(dataset, batch_size=len(dataset), collate_fn=scan_collate_fn, pin_memory=True, num_workers=6)
    for xyz, feature, indices, pts, y in traindlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            p, closs = model(xyz, feature, indices, pts)
            loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20) + closs
        loss.backward()
        # Gradient clipping in warmup
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

cur_id = args.run_id
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
errfile = f"output/log/{cur_id}/err.log"
logfile = open(logfile, "a", 1)
errfile = open(errfile, "a", 1)
sys.stdout = logfile
sys.stderr = errfile

print(f"Model: {model_type}")
print(f"Flash Attention: {getattr(dela_args, 'use_flash_attn_blocks', False)}")
if hasattr(dela_args, 'use_flash_attn_blocks') and dela_args.use_flash_attn_blocks:
    print(f"Flash Attention Layers: {dela_args.flash_attn_layers}")
print(f"Mamba2: {getattr(dela_args, 'run_mamba', False)}")
if hasattr(dela_args, 'run_mamba') and dela_args.run_mamba:
    print(f"Mamba2 Depth: {dela_args.mamba_depth}")
print(f"Gradient Clipping: {args.grad_clip}")

# Initialize wandb
wandb.init(
    project="DeLA-ScanNet",
    name=f"run_{cur_id}_{model_type}",
    config={
        "model_type": model_type,
        "use_flash_attn_blocks": getattr(dela_args, 'use_flash_attn_blocks', False),
        "flash_attn_layers": getattr(dela_args, 'flash_attn_layers', 0),
        "run_mamba": getattr(dela_args, 'run_mamba', False),
        "mamba_depth": getattr(dela_args, 'mamba_depth', []),
        "mamba_drop_path_rate": getattr(dela_args, 'mamba_drop_path_rate', 0.0),
        "grad_clip": args.grad_clip,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epoch,
        "warmup_epochs": warmup,
        "label_smoothing": ls,
        **vars(scan_args),
        **vars(dela_args),
    }
)

traindlr = DataLoader(ScanNetV2(scan_args, partition="train", loop=6), batch_size=batch_size, 
                      collate_fn=scan_collate_fn, shuffle=True, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=16)
testdlr = DataLoader(ScanNetV2(scan_args, partition="val", loop=1, train=False), batch_size=1,
                      collate_fn=scan_collate_fn, pin_memory=True, 
                      persistent_workers=True, num_workers=16)

step_per_epoch = len(traindlr)

model = ModelClass(dela_args).cuda()

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial = epoch * step_per_epoch, lr_min = lr/10000,
                                warmup_t=warmup*step_per_epoch, warmup_lr_init = lr/20)
scaler = GradScaler()
# if wish to continue from a checkpoint
resume = True
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(20)
ttls = util.AverageMeter() 
corls = util.AverageMeter() 
grad_norms = util.AverageMeter()  # Track gradient norms
best = 0
warmup_fn(model, ScanNetV2(scan_warmup_args, partition="train", loop=batch_size, warmup=True), args.grad_clip)
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    metric.reset()
    corls.reset()
    grad_norms.reset()  # Reset gradient norms
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
        mask = y != 20
        with autocast():
            p, closs = model(xyz, feature, indices, pts)
            loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20)
        metric.update(p.detach()[mask], y[mask])
        ttls.update(loss.item())
        corls.update(closs.item())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss + lam*closs).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_norms.update(grad_norm.item())  # Track gradient norm
        
        scaler.step(optimizer)
        scaler.update()
            
    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:")
    train_miou = metric.miou
    train_loss = ttls.avg
    train_closs = corls.avg
    train_grad_norm = grad_norms.avg
    wandb.log(
        {
            "epoch": i,
            "train_loss": train_loss,
            "train_closs": train_closs,
            "train_miou": train_miou,
            "learning_rate": optimizer.param_groups[0]["lr"],
        },
        step=i,
    )
    
    model.eval()
    metric.reset()
    val_ttls = util.AverageMeter()
    with torch.no_grad():
        for xyz, feature, indices, pts, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            y = y.cuda(non_blocking=True)
            mask = y != 20
            pts = pts.tolist()[::-1]
            with autocast():
                p = model(xyz, feature, indices, pts)
                loss = F.cross_entropy(p, y, label_smoothing=ls, ignore_index=20)
            metric.update(p[mask], y[mask])
            val_ttls.update(loss.item())
    
    metric.print("val:  ")
    val_miou = metric.miou
    val_macc = metric.macc
    val_acc = metric.acc
    val_loss = val_ttls.avg
    duration = time() - now
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
wandb.finish()
