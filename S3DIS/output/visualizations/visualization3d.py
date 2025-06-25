import torch
import matplotlib
matplotlib.use("Agg")          # falls du headless arbeitest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # sorgt für den 3D-Projection-Support
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.cutils import grid_subsampling

# --- 1) Lade den Sample ---
xyz, col, lbl = torch.load("data/s3dis/1_hallway_1.pt")

# --- 2) Subsample ---
grid_size = 0.04
ratio     = 2.5 / 14
indices   = grid_subsampling(xyz, grid_size, ratio)
sub_xyz   = xyz[indices]

# --- 3) Farbverlauf nach Einfüge-Index ---
order      = torch.arange(sub_xyz.shape[0])
order_norm = (order.float() / order.max()).numpy()

# --- 4) 3D-Scatterplot ---
fig = plt.figure(figsize=(8,8))
ax  = fig.add_subplot(111, projection="3d")
p   = ax.scatter(
    sub_xyz[:,0].numpy(), 
    sub_xyz[:,1].numpy(), 
    sub_xyz[:,2].numpy(),
    c=order_norm, 
    s=5, 
    cmap="viridis",
    depthshade=True
)

# Achsenbeschriftung und Farbskala
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
fig.colorbar(p, ax=ax, label="Einfüge-Index (0 → N)")
ax.set_title("Subsample-Reihenfolge (3D)")

# Speichern statt show
plt.savefig("subsample_plot_gridss_order_3d.png", dpi=300)
print("3D-Plot gespeichert unter subsample_plot_gridss_order_3d.png")
