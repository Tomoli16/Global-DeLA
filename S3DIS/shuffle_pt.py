import torch
from pathlib import Path
from config import raw_data_path, processed_data_path


def shuffle_pt_files(processed_data_dir: str, seed: int = None):
    """
    Lädt alle .pt-Dateien im Verzeichnis, permutiert die Punkte zufällig
    und speichert sie wieder.
    """
    processed_path = Path(processed_data_dir)
    pt_files = list(processed_path.glob("*.pt"))

    if seed is not None:
        torch.manual_seed(seed)

    for pt_file in pt_files:
        xyz, col, lbl = torch.load(pt_file)
        perm = torch.randperm(xyz.size(0))
        xyz = xyz[perm]
        col = col[perm]
        lbl = lbl[perm]
        torch.save((xyz, col, lbl), pt_file)
        print(f"Shuffled {pt_file.name}")

# Beispielaufruf:
shuffle_pt_files(processed_data_path, seed=42)
