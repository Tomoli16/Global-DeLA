import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.cutils import grid_subsampling

# --- 1) Lade einen Beispiel-Sample (Pfad anpassen) ---
xyz, col, lbl = torch.load("data/s3dis/1_hallway_1.pt")

# --- 2) Subsample mit deinem grid_size und ratio ---
grid_size = 0.04
ratio     = 2.5 / 14
indices   = grid_subsampling(xyz, grid_size, ratio)

sub_xyz   = xyz[indices]

# # Random permutation der Indizes für die Visualisierung
# sub_xyz   = sub_xyz[torch.randperm(sub_xyz.shape[0])]

# --- 3) Erzeuge einen Farb­ver­lauf nach Einfüge-Index ---
order     = torch.arange(sub_xyz.shape[0])
order_norm= (order.float() / order.max()).numpy()

# --- 4) Scatterplot in XY mit Farb­skala ---
plt.figure(figsize=(6,6))
plt.scatter(sub_xyz[:,0], sub_xyz[:,1], c=order_norm, s=5, cmap="viridis")
plt.colorbar(label="Einfüge-Index (0 → N)")
plt.title("Subsample-Reihenfolge (XY-Projektion)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.savefig("subsample_plot_gridss_order.png", dpi=300)  # Statt plt.show()
print("Plot gespeichert unter subsample_plot_gridss_order.png")
