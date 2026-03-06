"""
Stratified K-Fold Cross-Validation training script.

Used in Phase 3 to rigorously evaluate the tuned RvT model.
The model is reinitialized with random weights for each fold.

Usage:
    python scripts/train_kfold.py --model rvt --config configs/kfold_rvt.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model, HUGGINGFACE_MODELS
from utils.dataset import generate_df, ImageDataset, create_dataloaders
from utils.transforms import get_train_transform, get_val_test_transform

import pandas as pd
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_logits=False):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        logits = outputs.logits if use_logits else outputs
        _, preds = torch.max(logits, 1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()

    n = len(dataloader.dataset)
    return running_loss / n, running_corrects / n


def validate(model, dataloader, criterion, device, use_logits=False):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if use_logits else outputs
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

    n = len(dataloader.dataset)
    return running_loss / n, running_corrects / n


def build_optimizer(model, cfg):
    opt_name = cfg["optimizer"].lower()
    lr = cfg["lr"]
    wd = cfg.get("weight_decay", 0)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.9), weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg):
    sched_name = cfg.get("scheduler")
    if sched_name is None or sched_name == "null":
        return None
    if sched_name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size", 10), gamma=cfg.get("gamma", 0.5))
    elif sched_name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 100), eta_min=0)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation Training")
    parser.add_argument("--model", type=str, required=True, choices=["rvt", "cnn", "resnet50", "vit", "cvt"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build full dataset DataFrame
    data_cfg = cfg["data"]
    positive_df = generate_df(data_cfg["positive_dir"], label="POSITIVE")
    negative_df = generate_df(data_cfg["negative_dir"], label="NEGATIVE")
    all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

    # Encode labels for stratification
    labels_encoded = (all_df["Label"] == "POSITIVE").astype(int).values

    # K-Fold setup
    kfold_cfg = cfg["kfold"]
    n_folds = kfold_cfg["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=data_cfg["random_state"])

    train_transform = get_train_transform(data_cfg["image_size"])
    val_test_transform = get_val_test_transform(data_cfg["image_size"])

    use_logits = args.model in HUGGINGFACE_MODELS
    train_cfg = cfg["training"]

    fold_val_accs = []
    fold_val_losses = []

    print(f"\n{'='*60}")
    print(f"  Stratified {n_folds}-Fold Cross-Validation: {args.model.upper()}")
    print(f"{'='*60}\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, labels_encoded)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        train_df = all_df.iloc[train_idx].reset_index(drop=True)
        val_df = all_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = ImageDataset(train_df, transform=train_transform)
        val_dataset = ImageDataset(val_df, transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=data_cfg.get("num_workers", 2))
        val_loader = DataLoader(val_dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg.get("num_workers", 2))

        # Reinitialize model with fresh random weights for each fold
        model = build_model(args.model, cfg["model"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_scheduler(optimizer, train_cfg)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = f"{args.save_dir}/{args.model}_fold{fold + 1}_best.pth"

        for epoch in range(train_cfg["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_logits)
            val_loss, val_acc = validate(model, val_loader, criterion, device, use_logits)

            if scheduler:
                scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= train_cfg["patience"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Load best model for this fold and get final val metrics
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        final_val_loss, final_val_acc = validate(model, val_loader, criterion, device, use_logits)

        fold_val_accs.append(final_val_acc)
        fold_val_losses.append(final_val_loss)
        print(f"  Fold {fold + 1} Best → Val Loss: {final_val_loss:.4f} | Val Acc: {final_val_acc:.4f}")

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"  K-FOLD RESULTS SUMMARY")
    print(f"{'='*60}")
    for i in range(n_folds):
        print(f"  Fold {i + 1}: Acc = {fold_val_accs[i]:.4f}, Loss = {fold_val_losses[i]:.4f}")

    mean_acc = np.mean(fold_val_accs)
    std_acc = np.std(fold_val_accs)
    mean_loss = np.mean(fold_val_losses)
    std_loss = np.std(fold_val_losses)

    print(f"\n  Average Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Average Validation Loss:    {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
