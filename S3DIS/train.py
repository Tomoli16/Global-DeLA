import random, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys, math
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
from utils.ptv3_evaluator import SemSegEvaluator
import utils.util as util
from delasemseg import DelaSemSeg
from gridssmamba import GridSSMamba
from time import time, sleep
from config import s3dis_args, s3dis_warmup_args, dela_args, batch_size, learning_rate as lr, epoch, warmup, label_smoothing as ls
import wandb

torch.set_float32_matmul_precision("high")

def warmup_fn(model, dataset):
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
            loss = F.cross_entropy(p, y) + closs
        loss.backward()

cur_id = "02"
os.makedirs(f"output/log/{cur_id}", exist_ok=True)
os.makedirs(f"output/model/{cur_id}", exist_ok=True)
logfile = f"output/log/{cur_id}/out.log"
errfile = f"output/log/{cur_id}/err.log"
logfile = open(logfile, "a", 1)
errfile = open(errfile, "a", 1)
sys.stdout = logfile
sys.stderr = errfile
# 
print(r"base ")

# Initialize wandb
wandb.init(
    project="DeLA-S3DIS",
    name=f"run_{cur_id}",
    config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epoch,
        "warmup_epochs": warmup,
        "label_smoothing": ls,
        **vars(s3dis_args),
        **vars(dela_args),
    },
)

traindlr = DataLoader(S3DIS(s3dis_args, partition="!5", loop=30), batch_size=batch_size, # xyz, feature, indices, lbl
                      collate_fn=s3dis_collate_fn, shuffle=True, pin_memory=True, 
                      persistent_workers=True, drop_last=True, num_workers=16)
testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=1, train=False), batch_size=1,
                      collate_fn=s3dis_collate_fn, pin_memory=True, 
                      persistent_workers=True, num_workers=16)
print(len(traindlr))


step_per_epoch = len(traindlr)

model = DelaSemSeg(dela_args).cuda()
# model = GridSSMamba(dela_args).cuda()

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial = epoch * step_per_epoch, lr_min = lr/10000,
                                warmup_t=warmup*step_per_epoch, warmup_lr_init = lr/20)
scaler = GradScaler()
# if wish to continue from a checkpoint
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(13)
ttls = util.AverageMeter() 
corls = util.AverageMeter() 
best = 0
print("Starting warmup...")
warmup_fn(model, S3DIS(s3dis_warmup_args, partition="!5", loop=batch_size, warmup=True))
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    corls.reset()
    metric.reset()
    now = time()
    for xyz, feature, indices, pts, y in traindlr:
        lam = scheduler_step/(epoch*step_per_epoch)
        lam = 3e-3 ** lam * .25
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with autocast():
            p, closs = model(xyz, feature, indices, pts)
            loss = F.cross_entropy(p, y, label_smoothing=ls)
        metric.update(p.detach(), y)
        ttls.update(loss.item())
        corls.update(closs.item())

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss + closs*lam).backward()
        scaler.step(optimizer)
        scaler.update()

    
    
        
    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:")
    train_miou = metric.miou
    train_loss = ttls.avg
    train_closs = corls.avg
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

    metric.reset()


            
    evaluator = SemSegEvaluator(
        model=model,
        dataloader=testdlr,
        num_classes=13,
        ignore_index=100,
    )

    results = evaluator.evaluate() 
    
    
    
    # metric.print("val:  ")
    val_miou = results["mIoU"]
    val_macc = results["mAcc"]
    val_acc = results["allAcc"]
    wandb.log({
        "epoch": i,
        "val_miou": val_miou,
        "val_macc": val_macc,
        "val_acc": val_acc,
    }, step=i)
    print(f"duration: {time() - now}")
    cur = val_miou
    if best < cur:
        best = cur
        print("new best!")
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)
    
    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, scaler=scaler, start_epoch=i+1)
wandb.finish()
