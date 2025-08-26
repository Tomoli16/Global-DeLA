# Global DeLA

This repository contains the code for **Global DeLA**, a hybrid architecture for 3D point cloud semantic segmentation that augments the efficient DeLA backbone with global token mixing modules (FlashAttention and Mamba-2).  
We provide training and evaluation pipelines for **S3DIS** and **ScanNetV2**.

---

## 📦 Environment Setup

Clone the repository including submodules:

```bash
git clone --recursive git@github.com:Tomoli16/Global-DeLA.git
cd Global-DeLA
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

### S3DIS

Download S3DIS from [ETH CVG](https://cvg-data.inf.ethz.ch/s3dis/) (we use `Stanford3dDataset_v1.2_Aligned_Version`).
After download, set the correct `raw_data_path` in your config file and run:

```bash
python prepare_s3dis.py
```

This will preprocess the dataset into the required format.

### ScanNetV2

Follow the instructions from the official [ScanNet GitHub](https://github.com/ScanNet/ScanNet) to obtain the data.
After downloading, run:

```bash
python prepare_scannetv2.py
```

For additional details regarding dataset preparation and setup, please also check the [official DeLA repository](https://github.com/Matrix-ASC/DeLA/tree/main).

---

## ⚙️ Configuration

Training is fully configured through the `config/` files:

* For **S3DIS**: `S3DIS/config/`
* For **ScanNetV2**: `ScanNetV2/config/`

Each config specifies all hyperparameters (backbone, GTM module, serialization).
You can switch between predefined variants or define your own:

* **GDLA-Light**: Uses FlashAttention in the final stage.
* **GDLA-Heavy**: Uses Mamba-2 after the backbone.

Simply change the `model_type` field in the corresponding config.

---

## 🚀 Training

Run training with:

```bash
cd S3DIS
python train.py
```

or

```bash
cd ScanNetV2
python train.py
```

All arguments are passed via the config files.

---

## ✅ Testing

After training, you can evaluate a trained model checkpoint by running:

```bash
cd S3DIS
python test.py
```

or

```bash
cd ScanNetV2
python test.py
```

Again, all arguments and paths are handled via the config files.

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
