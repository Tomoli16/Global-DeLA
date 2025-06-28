import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np

def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

class SemSegEvaluator:
    def __init__(self, model, dataloader, num_classes, ignore_index=-1, class_names=None, device="cuda"):
        self.model = model
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.device = device
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, device=self.device)
        self.union = torch.zeros(self.num_classes, device=self.device)
        self.target = torch.zeros(self.num_classes, device=self.device)
        self.losses = []

    def evaluate(self, loss_fn=None, use_origin_mapping=False):
        self.reset()
        self.model.eval()
        with torch.no_grad():
            for xyz, feature, indices, pts, y in self.dataloader:
                inputs = xyz.to(self.device)
                if feature is not None:
                    feature = feature.to(self.device)
                
                target = y.to(self.device)
                indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
                pts = pts.tolist()[::-1]
                with autocast():
                    logits = self.model(inputs, feature, indices, pts)


                preds = logits.argmax(dim=1)


                inter, uni, tgt = intersection_and_union_gpu(
                    preds, target, self.num_classes, self.ignore_index
                )
                self.intersection += inter
                self.union += uni
                self.target += tgt

        return self.compute()

    def compute(self):
        iou = self.intersection / (self.union + 1e-10)
        acc = self.intersection / (self.target + 1e-10)
        m_iou = iou.mean().item()
        m_acc = acc.mean().item()
        all_acc = self.intersection.sum().item() / (self.target.sum().item() + 1e-10)
        avg_loss = np.mean(self.losses) if self.losses else None

        print(f"[Eval] mIoU: {m_iou:.4f} | mAcc: {m_acc:.4f} | allAcc: {all_acc:.4f}")
        for i in range(self.num_classes):
            print(f"  Class {i} ({self.class_names[i]}): IoU={iou[i]:.4f} | Acc={acc[i]:.4f}")

        return {
            "mIoU": m_iou,
            "mAcc": m_acc,
            "allAcc": all_acc,
            "class_IoU": iou.cpu().tolist(),
            "class_Acc": acc.cpu().tolist(),
            "loss": avg_loss
        }
