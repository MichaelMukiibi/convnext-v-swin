import sys
import subprocess

required_packages = ["medmnist", "transformers", "scikit-learn", "wandb", "torch", "torchvision"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import medmnist
from medmnist import INFO
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from transformers import AutoModelForImageClassification
import wandb

# ------------------------------------------------------------------------------
# 1. Pipeline Functions
# ------------------------------------------------------------------------------

def get_dataloaders(dataset_flag='pneumoniamnist', batch_size=32):
    """Loads PneumoniaMNIST and transforms images to 3-channel 224x224 RGB tensors."""
    info = INFO[dataset_flag]
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = DataClass(split='train', transform=transform, download=True)
    val_ds = DataClass(split='val', transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, num_classes


def build_feature_extractor(checkpoint_name, num_classes):
    """Initializes model and freezes base backbone parameters."""
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # Freeze base feature extractor parameters
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Executes a single training pass over the dataset."""
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.squeeze(1).long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model and computes Loss, Accuracy, AUC-ROC, Sensitivity, and Specificity."""
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.squeeze(1).long().to(device)

        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        running_loss += loss.item() * images.size(0)
        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels.cpu())

    val_loss = running_loss / len(dataloader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Probabilities and Predictions
    probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()
    preds = torch.argmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()

    # Standard Metrics
    accuracy = accuracy_score(labels_np, preds)
    auc_roc = roc_auc_score(labels_np, probs)

    # Confusion Matrix: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(labels_np, preds).ravel()

    # Sensitivity (True Positive Rate / Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return val_loss, accuracy, auc_roc, sensitivity, specificity


def run_experiment(model_alias, checkpoint, args, train_loader, val_loader, num_classes, device):
    """Runs end-to-end feature extraction training for a specified architecture."""
    print(f"\n================ Starting Experiment: {model_alias} ================")
    
    # 1. Setup W&B Logging
    use_wandb = bool(args.wandb_key)
    if use_wandb:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project=args.wandb_project,
            name=f"{model_alias}-feature-extraction",
            config={
                "model_alias": model_alias,
                "checkpoint": checkpoint,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "dataset": "pneumoniamnist"
            },
            reinit=True
        )

    # 2. Build Model & Optimizer
    model = build_feature_extractor(checkpoint, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_sens, val_spec = evaluate(model, val_loader, criterion, device)

        print(
            f"[{model_alias}] Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | "
            f"Sens: {val_sens:.4f} | Spec: {val_spec:.4f}"
        )   

        if use_wandb:
            wandb.log({
                f"{model_alias}/train_loss": train_loss,
                f"{model_alias}/val_loss": val_loss,
                f"{model_alias}/val_accuracy": val_acc,
                f"{model_alias}/val_auc_roc": val_auc,
                f"{model_alias}/val_sensitivity": val_sens,
                f"{model_alias}/val_specificity": val_spec,
                "epoch": epoch
            })

    if use_wandb:
        wandb.finish()

    return model

# ------------------------------------------------------------------------------
# 2. Main Entry Point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline for ConvNeXt vs Swin")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for classification head")
    parser.add_argument("--wandb_key", type=str, default="", help="Weights & Biases API Key")
    parser.add_argument("--wandb_project", type=str, default="medmnist-convnext-vs-swin", help="W&B Project Name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target Device: {device}")

    # Load Data
    train_loader, val_loader, num_classes = get_dataloaders('pneumoniamnist', batch_size=args.batch_size)

    # Models Registry
    models_to_benchmark = {
        "ConvNeXt": "facebook/convnext-tiny-224",
        "Swin": "microsoft/swin-tiny-patch4-window7-224"
    }

    # Sequentially Train Both Models safely without variable leakage
    for alias, checkpoint in models_to_benchmark.items():
        run_experiment(
            model_alias=alias,
            checkpoint=checkpoint,
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=device
        )

if __name__ == "__main__":
    main()