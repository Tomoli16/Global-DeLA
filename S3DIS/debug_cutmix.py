import torch
from s3dis import cutmix_pointcloud_torch  # oder wie deine Funktion heißt
import random
import math
import matplotlib.pyplot as plt


def make_dummy(N, C=6):
    # Erzeuge zufällige xyz im [0,1]³ und zufällige Features+Labels
    xyz  = torch.rand(N, 3)
    feat = torch.rand(N, C)
    lbl  = torch.randint(0, 10, (N,))
    return xyz, feat, lbl

def get_scene(name):
    xyz, col, lbl = torch.load(name)
    return xyz, col, lbl

def _augment_xyz(xyz: torch.Tensor) -> torch.Tensor:
        """
        Rotation um Hochachse, leichte Skalierung, gaussches Jitter
        und Rezentrieren der Punkte.
        """
        # 1) Zufälliger Winkel und Skalierung
        angle = random.random() * 2 * math.pi
        scale = random.uniform(0.8, 1.2)
        cos, sin = math.cos(angle), math.sin(angle)
        rotmat = torch.tensor([
            [cos,  sin,  0],
            [-sin, cos,  0],
            [0,    0,    1],
        ], device=xyz.device, dtype=xyz.dtype) * scale

        # 2) Rotation und Skalierung
        xyz = xyz @ rotmat

        # 3) Gaussches Rauschen
        xyz += torch.empty_like(xyz).normal_(std=0.005)

        # 4) Rezentrieren
        min_vals = xyz.min(dim=0, keepdim=True)[0]
        xyz = xyz - min_vals

        return xyz

if __name__ == "__main__":
    print(f"\n--- Test mit ganzen Punkwolken---")
    # xyz1, f1, l1 = make_dummy(N)
    # xyz2, f2, l2 = make_dummy(N)
    xyz1, f1, l1 = get_scene("data/s3dis/1_hallway_1.pt")
    xyz2, f2, l2 = get_scene("data/s3dis/1_office_10.pt")
    # try:
    #     xyz1 = _augment_xyz(xyz1)
    #     xyz2 = _augment_xyz(xyz2)
    #     print("→ OK: xyz1:", xyz1.shape, "xyz2:", xyz2.shape)
    # except Exception as e:
    #     print("!!! Exception in _augment_xyz:", repr(e))
    try:
        xyz_new, f_new, l_new = cutmix_pointcloud_torch(
            xyz1, f1, l1,
            xyz2, f2, l2,
            beta=1.0
        )
        print("→ OK:", xyz_new.shape, f_new.shape, l_new.shape)
        # Visualisierung
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xyz_new[:, 0].cpu(), xyz_new[:, 1].cpu(), xyz_new[:, 2].cpu(), s=1)
        ax.set_title(f"CutMix Result")
        plot_name = f"cutmix_result.png"
        plt.savefig(plot_name, dpi=300)
        print(f"3D-Plot gespeichert unter {plot_name}")

    except Exception as e:
        print("!!! Exception:", repr(e))
        # für PyTorch‑Autograd‑Bugs:
        import traceback; traceback.print_exc()
