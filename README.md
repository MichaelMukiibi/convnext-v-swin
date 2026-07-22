# ConvNeXt vs. Swin Transformer on MedMNIST

A modular benchmarking framework for comparative transfer learning evaluation between **ConvNeXt-Tiny** and **Swin-Tiny** architectures using the **PneumoniaMNIST** dataset.

---

## 📌 Overview

This project provides an automated pipeline for performing **feature extraction** (linear probing) on medical imaging data (`pneumoniamnist`). It evaluates two state-of-the-art vision architectures using identical preprocessing and optimization parameters:

* **ConvNeXt (`facebook/convnext-tiny-224`):** A modernized, pure convolutional architecture adopting Vision Transformer design choices.
* **Swin Transformer (`microsoft/swin-tiny-patch4-window7-224`):** A hierarchical Vision Transformer utilizing shifted local windows for linear computational complexity.

---

## 🛠️ Setup & Installation

Ensure you have Python 3.8+ installed along with GPU acceleration support.

```bash
pip install torch torchvision transformers medmnist scikit-learn wandb
```

## Usage

Execute the pipeline via command line or inside a Colab environment using `colab run`.

## Standard Execution

```bash
python train.py --epochs 5 --batch_size 32 --lr 1e-3
```

## Execution with Weights & Biases Logging

To log loss curves, Accuracy, and AUC-ROC metrics to Weights & Biases:

```.env
WANDB_API_KEY=wandb_key_here
```

```bash

source .env

python train.py \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-3 \
  --wandb_key $WANDB_API_KEY \
  --wandb_project medmnist-convnext-vs-swin

```

## Google Colab CLI Execution (colab run)

```bash

colab run --gpu t4 -s medmnist-benchmark train.py \
  --epochs 5 \
  --batch_size 32 \
  --wandb_key $WANDB_API_KEY

  ```

  