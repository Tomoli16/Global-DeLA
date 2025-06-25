import torch
import plotly.graph_objects as go
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.cutils import grid_subsampling
from utils.transforms import serialization


# --- 1) Sample laden ---
xyz, col, lbl = torch.load("data/s3dis/1_hallway_1.pt")

# --- 2) Subsample ---
grid_size = 0.04
ratio     = 2.5 / 14
indices   = grid_subsampling(xyz, grid_size, ratio)
sub_xyz   = xyz[indices]

# # Random permutation der Indizes für die Visualisierung
# sub_xyz   = sub_xyz[torch.randperm(sub_xyz.shape[0])]


# --- 3) Flatten für die Serialization ---
xyz_flat = sub_xyz.unsqueeze(0)                # (N,3)
x_flat   = sub_xyz.clone().unsqueeze(0)        # Features; hier einfach die Koordinaten
pts      = [sub_xyz.shape[0]]     # Liste mit der Punktzahl der Szene

# --- 4) Serialisierung aufrufen ---
#    inverse_order gibt für jeden Punkt den Serialisierungs-Index
sub_xyz, x_ser, _ = serialization(
    xyz_flat,
    x_flat,
    order="hilbert",        # z-Raster-Reihenfolge
    grid_size=grid_size
)
sub_xyz = sub_xyz.squeeze(0)  # zurück zu (N,3)


# --- 3) Farbverlauf nach Einfüge-Index ---
order      = torch.arange(sub_xyz.shape[0])
order_norm = (order.float() / order.max()).numpy()

# --- 4) Interaktives Plotly-Figure erstellen ---
fig = go.Figure(data=[
    go.Scatter3d(
        x=sub_xyz[:,0].numpy(),
        y=sub_xyz[:,1].numpy(),
        z=sub_xyz[:,2].numpy(),
        mode='markers',
        marker=dict(
            size=2,
            color=order_norm,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Einfüge-Index")
        )
    )
])

# Layout anpassen
fig.update_layout(
    scene = dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),
    margin=dict(r=10, l=10, b=10, t=10),
    width=800,
    height=800,
    title="Subsample-Reihenfolge (interaktiv)"
)

# Als HTML speichern
output_path = "interactive_subsample_random.html"
fig.write_html(output_path, include_plotlyjs='cdn')
print(f"Interaktive 3D-Ansicht gespeichert unter {output_path}")
