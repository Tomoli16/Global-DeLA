# Global DeLA

This repository contains the code for **Global DeLA**, a hybrid architecture for 3D point cloud semantic segmentation that augments the efficient DeLA backbone with global token mixing modules (FlashAttention and Mamba-2).  
We provide training and evaluation pipelines for **S3DIS** and **ScanNetV2**.

---

## 📦 Environment Setup

Clone the repository including submodules:

```bash
git clone --recursive <repo-url>
cd <repo-name>
````

If you already cloned without `--recursive`, initialize the submodule manually:

```bash
git submodule update --init --recursive modules/mamba
```

Set up the environment by running:

```bash
source setup.sh
```

This will install all dependencies (PyTorch, FlashAttention, Mamba, etc.).

---

## 🗂 Datasets

* **S3DIS**: Place the processed dataset under `S3DIS/data/`.
* **ScanNetV2**: Place the processed dataset under `ScanNetV2/data/`.

Refer to the official dataset websites for download instructions.

---

## ⚙️ Configuration

Training is fully configured through the `config/` files:

* For **S3DIS**: `S3DIS/config/`
* For **ScanNetV2**: `ScanNetV2/config/`

Each config specifies all hyperparameters (backbone, GTM module, serialization, augmentation, optimizer).
You can switch between predefined variants or define your own:

* **GDLA-Light**: Uses FlashAttention in the final stage.
* **GDLA-Heavy**: Uses Mamba-2 after the backbone.

Simply change the `model` field in the corresponding config.

**Example (YAML):**

```yaml
model: GDLA-Light
dataset: S3DIS
max_points: 30000
batch_size: 8
epochs: 100
optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.05
scheduler:
  type: cosine
  warmup_epochs: 10
```

---

## 🚀 Training

Run training with:

```bash
cd S3DIS
python train.py --config config/<your_config>.yaml
```

or

```bash
cd ScanNetV2
python train.py --config config/<your_config>.yaml
```

All arguments are passed via the config files.

---

## 📊 Results

We provide configs to reproduce the main results reported in the paper:

* **S3DIS**:

  * GDLA-Light (+1 mIoU over DeLA baseline, faster convergence)
  * GDLA-Heavy (+4.8 mIoU over official DeLA baseline)

* **ScanNetV2**:

  * GDLA-Light (+0.1 mIoU over DeLA baseline, faster convergence)

---

## 📌 Notes

* Make sure to load the **Mamba submodule** (`modules/mamba`) before running training.
* All results in the paper were obtained with the configs included in this repository.
* You can modify any config file to explore custom backbone depths, feature dims, or GTM modules.

---

## 📜 Citation

If you use this code in your research, please cite:

```
@misc{your2025globaldela,
  author       = {Your Name},
  title        = {Global DeLA: Efficient Global Token Mixing for 3D Point Clouds},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {<repo-url>}
}
```

```

---

👉 Soll ich dir auch gleich eine **sekundäre Trainingsanleitung für Slurm/HPC** (Jobscript mit `srun`/`sbatch`) in die README einbauen, da du ja auch auf der H100/Capella trainierst?
```
