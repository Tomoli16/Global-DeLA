import torch
import matplotlib.pyplot as plt
from utils.cutils import grid_subsampling

# --- 1) Lade einen Beispiel-Sample (Pfad anpassen) ---
xyz, col, lbl = torch.load("path/to/sample.pt")

# --- 2) Subsample mit deinem grid_size und ratio ---
grid_size = 0.04
ratio     = 2.5 / 14
indices   = grid_subsampling(xyz, grid_size, ratio)

sub_xyz   = xyz[indices]

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
plt.show()
