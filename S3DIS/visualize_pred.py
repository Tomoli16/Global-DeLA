import torch
import matplotlib.pyplot as plt
import numpy as np
from delasemseg import DelaSemSeg
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys, math
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from config import s3dis_args, s3dis_warmup_args, dela_args, batch_size, learning_rate as lr, epoch, warmup, label_smoothing as ls




class PointSegmentationPipeline:
    """
    Pipeline to load a point cloud scene, predict semantic labels, and visualize both
    ground truth and predicted segmentation.
    """
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: segmentation model with forward(xyz, features, indices, pts_list)
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device

    def predict(self, dataset):
        """
        Load a .pt scene, run model, and return xyz coordinates, ground truth labels,
        and predicted labels.
        
        Args:
            pt_file: path to the .pt file containing (xyz, features, labels)
        Returns:
            xyz: numpy array shape (N, 3)
            gt: numpy array shape (N,)
            pred: numpy array shape (N,)
        """
        # Load point cloud scene
        for xyz, features, indices, pts, labels in dataset:
            break
        xyz = xyz.to(self.device)
        features = features.to(self.device)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        labels = labels.cuda(non_blocking=True)
                
        self.model.eval()
        with torch.no_grad():
            logits = self.model(xyz, features, indices, pts)
            print("logits.shape:", logits.shape)
            # logits shape [num_classes, N]
            pred = logits.argmax(dim=1).cpu().numpy()
        
        return xyz.cpu().numpy(), labels.cpu().numpy(), pred

    def visualize(self, xyz, gt, pred, s=1, figsize=(12, 6)):
        """
        Visualize ground truth and predicted segmentation side by side.
        
        Args:
            xyz: (N, 3) coordinate array
            gt:  (N,) ground truth labels
            pred:(N,) predicted labels
            s: scatter point size
        """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=gt, s=s)
        ax1.set_title("Ground Truth")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=pred, s=s)
        ax2.set_title("Prediction")
        ax2.axis('off')
        
        filename = "visualized_segmentation.png"
        plt.savefig(filename, dpi=300)
        print("Visualization saved to", filename)

    def run(self, dataset):
        """
        Load, predict, and visualize a single scene.
        
        Args:
            pt_file: path to the .pt scene file
        """
        xyz, gt, pred = self.predict(dataset)
        self.visualize(xyz, gt, pred)

# Example usage:
# pipeline = PointSegmentationPipeline(model, device='cuda')
# pipeline.run("data/s3dis/1_hallway_1.pt")
if __name__ == "__main__":

    dataset = DataLoader(S3DIS(s3dis_args, partition="2", loop=1, train=False), batch_size=1,
        collate_fn=s3dis_collate_fn, pin_memory=True, 
        persistent_workers=True, num_workers=16)
    scene_idx = 5
    model = DelaSemSeg(dela_args).cuda()
    util.load_state("output/model/03/last.pt", model=model)
    pipeline = PointSegmentationPipeline(model, device='cuda')
    pipeline.run(dataset)