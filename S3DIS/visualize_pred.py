import torch
import matplotlib.pyplot as plt
import numpy as np
from s3dis import S3DIS, s3dis_test_collate_fn
from torch.utils.data import DataLoader
import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from config import s3dis_args, dela_args, processed_data_path
from torch.cuda.amp import autocast
import importlib
from typing import List, Set

# S3DIS class names in order (0..12)
S3DIS_LABELS: List[str] = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
]

def parse_exclude_labels(arg: str) -> Set[int]:
    """Parse comma-separated labels (names or indices) into a set of class ids."""
    if not arg:
        return set()
    ids: Set[int] = set()
    name_to_id = {n.lower(): i for i, n in enumerate(S3DIS_LABELS)}
    for token in arg.split(','):
        t = token.strip()
        if not t:
            continue
        # try int
        try:
            ids.add(int(t))
            continue
        except ValueError:
            pass
        # try name
        i = name_to_id.get(t.lower())
        if i is not None:
            ids.add(i)
        else:
            print(f"Warning: unknown label '{t}', expected one of {S3DIS_LABELS} or indices 0-12")
    return ids

# Optional Open3D import
try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

# Optional Plotly import for HTML export
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False




class PointSegmentationPipeline:
    """
    Pipeline to load a specific S3DIS scene, predict semantic labels using test sampling,
    compute mIoU for that scene, and visualize full-scene ground truth vs prediction.
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def eval_scene(self, loader, loop, full_xyz, full_lbl):
        """
        Accumulate predictions over 'loop' picks and compute per-scene metrics.
        Returns accumulated logits [N_full, C].
        """
        self.model.eval()
        cum = 0
        cnt = 0
        metric = util.Metric(num_classes=13)
        with torch.no_grad():
            for xyz, feature, indices, nn, y in loader:
                xyz = xyz.to(self.device, non_blocking=True)
                feature = feature.to(self.device, non_blocking=True)
                indices = [ii.to(self.device, non_blocking=True).long() for ii in indices[::-1]]
                nn = nn.to(self.device, non_blocking=True).long()
                with autocast():
                    p = self.model(xyz, feature, indices)
                cum = cum + p[nn] if isinstance(cum, torch.Tensor) else p[nn]
                cnt += 1
                if cnt % loop == 0:
                    y = y.to(self.device, non_blocking=True)
                    metric.update(cum, y)
                    # Compute mIoU once for the scene
                    metric.calc()
                    return cum, metric
        return cum, metric

    def visualize(self, xyz, gt, pred, s=1, figsize=(12, 6)):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=gt, s=s)
        ax1.set_title("Ground Truth")
        ax1.axis('off')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=pred, s=s)
        ax2.set_title("Prediction")
        ax2.axis('off')

        filename = "visualized_segmentation.png"
        plt.savefig(filename, dpi=300)
        print("Visualization saved to", filename)

    def visualize_open3d(self, xyz, labels, point_size: float = 2.0, title: str = "Open3D Viewer"):
        """Interactive Open3D point cloud viewer colored by labels."""
        if not HAS_O3D:
            print("Open3D not available. Please install 'open3d'.")
            return

        # S3DIS 13-class palette (RGB 0-255), normalized to [0,1]
        palette = np.array([
            [152, 223, 138],  # ceiling
            [174, 199, 232],  # floor
            [31,  119, 180],  # wall
            [255, 187, 120],  # beam
            [188, 189, 34],   # column
            [140,  86,  75],  # window
            [255, 152, 150],  # door
            [214,  39,  40],  # table
            [197, 176, 213],  # chair
            [148, 103, 189],  # sofa
            [196, 156, 148],  # bookcase
            [23,  190, 207],  # board
            [247, 182, 210],  # clutter
        ], dtype=np.float32) / 255.0

        labels = np.asarray(labels).astype(np.int64)
        colors = palette[labels % palette.shape[0]]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=800, visible=True)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = float(point_size)
        opt.show_coordinate_frame = False
        vis.update_renderer()
        vis.run()
        vis.destroy_window()

    def visualize_plotly_html(self, xyz, labels, filename: str = None, point_size: float = 2.0, title: str = "Scene"):
        """Save interactive scene as standalone HTML using Plotly."""
        if not HAS_PLOTLY:
            print("Plotly not available. Please install 'plotly' to enable HTML export.")
            return

        xyz = np.asarray(xyz)
        labels = np.asarray(labels).astype(np.int64)
        if xyz.size == 0:
            print("Empty point set; skipping HTML export.")
            return

        # Palette for 13 classes
        palette = np.array([
            [152, 223, 138], [174, 199, 232], [31, 119, 180], [255, 187, 120],
            [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40],
            [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207],
            [247, 182, 210]
        ], dtype=np.float32) / 255.0
        colors = palette[labels % palette.shape[0]]

        # Convert colors to hex for Plotly
        colors_hex = [
            '#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors
        ]

        fig = go.Figure(data=[
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode='markers',
                marker=dict(size=point_size, color=colors_hex, opacity=1.0),
                hoverinfo='none'
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
            margin=dict(l=0, r=0, t=40, b=0),
            template='plotly_white'
        )

        out = filename or "scene_view.html"
        fig.write_html(out, include_plotlyjs='cdn', full_html=True)
        print(f"Interactive HTML saved to {out}")

# Example usage:
# pipeline = PointSegmentationPipeline(model, device='cuda')
# pipeline.run("data/s3dis/1_hallway_1.pt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize S3DIS scene with mIoU and optional Open3D/HTML viewers")
    parser.add_argument("--open3d", action="store_true", help="Open interactive Open3D viewer")
    parser.add_argument("--color", choices=["pred", "gt"], default="pred", help="Which labels to color by in the viewer")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size for Open3D viewer")
    parser.add_argument("--scene", type=str, default=None, help="Override scene filename, e.g., 1_office_21.pt")
    parser.add_argument("--weights", type=str, default="output/model/default_mamba_clean2/best.pt", help="Path to model weights")
    parser.add_argument("--baseline", action="store_true", help="Use baseline model (delasemseg_baseline.py) and pretrained/best.pt by default")
    parser.add_argument("--html", action="store_true", help="Export interactive HTML view (Plotly)")
    parser.add_argument("--html-file", type=str, default="scene_view.html", help="Output HTML filename")
    parser.add_argument("--no-ceiling", action="store_true", help="Exclude ceiling points (label 0) from viewer/HTML")
    parser.add_argument("--max-points", type=int, default=30000, help="Max points to subsample for metrics/visualization")
    parser.add_argument("--html-gt", action="store_true", help="Also export a ground-truth HTML view")
    parser.add_argument("--html-gt-file", type=str, default="scene_view_gt.html", help="Ground-truth HTML filename")
    parser.add_argument("--exclude-labels", type=str, default="", help="Comma-separated labels (names or ids) to exclude, e.g., 'clutter,beam' or '12,3'")
    parser.add_argument("--crop-top-percent", type=float, default=0.0, help="Crop top X percent of Z-range (0-100) to hide ceiling fixtures")
    args = parser.parse_args()

    # Target scene path
    scene_name = args.scene if args.scene else "5_office_34.pt"
    scene_path = processed_data_path / scene_name

    if not scene_path.exists():
        print(f"Scene file not found: {scene_path}")
        sys.exit(1)

    # Configure dataset to load only the target scene using test sampling
    loop = 12  # number of picks to cover full scene (as in test.py)
    ds = S3DIS(s3dis_args, partition="1", loop=loop, train=False, test=True)
    # Override to single scene
    ds.paths = [scene_path]
    ds.datas = [torch.load(scene_path)]

    # DataLoader for test
    loader = DataLoader(ds, batch_size=1, collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=0)

    # Select model implementation
    if args.baseline:
        model_mod = importlib.import_module('delasemseg_baseline')
        default_attn_weights = "output/model/default_mamba_clean2/best.pt"
        # If user didn't pass custom weights, use baseline default
        weights_path = args.weights if args.weights != default_attn_weights else "pretrained/best.pt"
    else:
        model_mod = importlib.import_module('global_dela')
        weights_path = args.weights

    ModelClass = getattr(model_mod, 'DelaSemSeg')
    model = ModelClass(dela_args).cuda()
    util.load_state(weights_path, model=model)

    # Full scene xyz and labels for visualization/metrics
    full_xyz, _, full_lbl = ds.datas[0]

    # Run evaluation on the single scene
    pipeline = PointSegmentationPipeline(model, device='cuda')
    cum_logits, metric = pipeline.eval_scene(loader, loop, full_xyz, full_lbl)

    # Subsample to a maximum number of points for metric and visualization
    max_pts = int(args.max_points) if hasattr(args, 'max_points') else 30000
    N = cum_logits.shape[0]
    device = cum_logits.device
    if N > max_pts:
        idx = torch.randperm(N, device=device)[:max_pts]
    else:
        idx = torch.arange(N, device=device)

    # Compute safe per-scene mIoU on subsample over present classes only
    metric_sub = util.Metric(num_classes=13)
    labels_gpu = (full_lbl.cuda() if torch.is_tensor(full_lbl) else torch.as_tensor(full_lbl, device=device)).long()
    metric_sub.update(cum_logits[idx], labels_gpu[idx])
    inter = metric_sub.intersection.float()
    union = metric_sub.union.float()
    mask = union > 0
    miou_scene = (inter[mask] / union[mask]).mean().item() if mask.any() else float('nan')

    used = idx.numel()
    print(f"Scene {scene_name} mIoU (on {used} pts): {miou_scene:.4f}")

    # Build predicted labels and arrays for visualization on the same subset
    pred = cum_logits[idx].argmax(dim=1).cpu().numpy()
    gt_full = full_lbl.cpu().numpy() if torch.is_tensor(full_lbl) else np.asarray(full_lbl)
    xyz_full = full_xyz.cpu().numpy() if torch.is_tensor(full_xyz) else np.asarray(full_xyz)
    idx_np = idx.cpu().numpy()
    gt = gt_full[idx_np]
    xyz = xyz_full[idx_np]

    # Visualize full-scene GT vs Prediction (static)
    pipeline.visualize(xyz, gt, pred)

    # Optional Open3D interactive viewer
    if args.open3d:
        chosen_labels = pred if args.color == "pred" else gt
        xyz_vis = xyz.copy()
        labels_vis = chosen_labels.copy()
        if args.no_ceiling:
            mask = (labels_vis != 0)
            if mask.any():
                xyz_vis = xyz_vis[mask]
                labels_vis = labels_vis[mask]
        # Exclude specific labels
        excl_ids = parse_exclude_labels(args.exclude_labels)
        if excl_ids:
            mask = ~np.isin(labels_vis, np.array(list(excl_ids)))
            if mask.any():
                xyz_vis = xyz_vis[mask]
                labels_vis = labels_vis[mask]
        # Crop top Z percent
        p = float(args.crop_top_percent)
        if p > 0:
            z = xyz_vis[:, 2]
            zmin, zmax = z.min(), z.max()
            thresh = zmin + (zmax - zmin) * (1 - p / 100.0)
            mask = z < thresh
            if mask.any():
                xyz_vis = xyz_vis[mask]
                labels_vis = labels_vis[mask]
        if not HAS_O3D:
            print("Open3D not installed. pip install open3d to enable viewer.")
        elif os.environ.get("DISPLAY") is None and sys.platform.startswith("linux"):
            print("No DISPLAY found; skipping interactive Open3D viewer on headless Linux.")
        else:
            pipeline.visualize_open3d(xyz_vis, labels_vis, point_size=args.point_size, title=f"{scene_name} ({args.color})")

    # Optional Plotly HTML export (works headless)
    if args.html:
        chosen_labels = pred if args.color == "pred" else gt
        xyz_vis = xyz.copy()
        labels_vis = chosen_labels.copy()
        if args.no_ceiling:
            mask = (labels_vis != 0)
            if mask.any():
                xyz_vis = xyz_vis[mask]
                labels_vis = labels_vis[mask]
        # Exclude specific labels
        excl_ids = parse_exclude_labels(args.exclude_labels)
        if excl_ids:
            mask = ~np.isin(labels_vis, np.array(list(excl_ids)))
            if mask.any():
                xyz_vis = xyz_vis[mask]
                labels_vis = labels_vis[mask]
        # Crop top Z percent
        p = float(args.crop_top_percent)
        if p > 0:
            z = xyz_vis[:, 2]
            zmin, zmax = z.min(), z.max()
            thresh = zmin + (zmax - zmin) * (1 - p / 100.0)
            mask = z < thresh
            if mask.any():
                xyz_vis = xyz_vis[mask]
                labels_vis = labels_vis[mask]
        pipeline.visualize_plotly_html(xyz_vis, labels_vis, filename=args.html_file, point_size=args.point_size, title=f"{scene_name} ({args.color})")

        # Optionally also export GT view (apply same filters)
        if args.html_gt:
            xyz_gt = xyz.copy()
            labels_gt = gt.copy()
            if args.no_ceiling:
                mask_gt = (labels_gt != 0)
                if mask_gt.any():
                    xyz_gt = xyz_gt[mask_gt]
                    labels_gt = labels_gt[mask_gt]
            # Exclude specific labels for GT as well
            excl_ids_gt = parse_exclude_labels(args.exclude_labels)
            if excl_ids_gt:
                mask_gt = ~np.isin(labels_gt, np.array(list(excl_ids_gt)))
                if mask_gt.any():
                    xyz_gt = xyz_gt[mask_gt]
                    labels_gt = labels_gt[mask_gt]
            # Crop top Z percent for GT
            p_gt = float(args.crop_top_percent)
            if p_gt > 0 and xyz_gt.size > 0:
                z = xyz_gt[:, 2]
                zmin, zmax = z.min(), z.max()
                thresh = zmin + (zmax - zmin) * (1 - p_gt / 100.0)
                mask_gt = z < thresh
                if mask_gt.any():
                    xyz_gt = xyz_gt[mask_gt]
                    labels_gt = labels_gt[mask_gt]
            pipeline.visualize_plotly_html(xyz_gt, labels_gt, filename=args.html_gt_file, point_size=args.point_size, title=f"{scene_name} (gt)")