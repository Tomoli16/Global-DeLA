import torch
import numpy as np
from delasemseg import DelaSemSeg
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import utils.util as util
from config import s3dis_args, dela_args

import plotly.graph_objs as go

class PointSegmentationPipeline:
    """
    Interactive 3D visualization pipeline for point cloud semantic segmentation
    that saves the result as an HTML file for offline viewing.
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
        # get first batch
        for xyz, features, indices, pts, labels in dataset:
            break
        xyz = xyz.to(self.device)
        features = features.to(self.device)
        indices = [ii.to(self.device).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        labels = labels.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(xyz, features, indices, pts)
            pred = logits.argmax(dim=1).cpu().numpy()

        return xyz.cpu().numpy(), labels.cpu().numpy(), pred

    def visualize_and_save(self, xyz, gt, pred, filename="segmentation.html", size=2):
        """
        Create interactive Plotly 3D scatter for ground truth and prediction,
        and save it to an HTML file.
        
        Args:
            xyz: (N,3) coordinate array
            gt:  (N,) ground truth labels
            pred:(N,) predicted labels
            filename: path for the output HTML file
            size: marker size
        """
        trace_gt = go.Scatter3d(
            x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
            mode='markers',
            marker=dict(size=size, color=gt, colorscale='Viridis'),
            name='Ground Truth'
        )
        trace_pred = go.Scatter3d(
            x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
            mode='markers',
            marker=dict(size=size, color=pred, colorscale='Viridis'),
            name='Prediction'
        )
        fig = go.Figure(data=[trace_gt, trace_pred])
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            title='Semantic Segmentation (GT vs Pred)',
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Save as standalone HTML
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Interactive plot saved to {filename}")

    def run(self, dataset, out_html="segmentation.html"):
        xyz, gt, pred = self.predict(dataset)
        self.visualize_and_save(xyz, gt, pred, filename=out_html)

if __name__ == "__main__":
    dataset = DataLoader(
        S3DIS(s3dis_args, partition="2", loop=1, train=False),
        batch_size=1,
        collate_fn=s3dis_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=8
    )
    model = DelaSemSeg(dela_args).cuda()
    util.load_state("output/model/03/best.pt", model=model)
    pipeline = PointSegmentationPipeline(model, device='cuda')
    pipeline.run(dataset, out_html="visualized_segmentation.html")
