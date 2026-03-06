"""
Train a model using a config file.

Supports all models (RvT, CNN, ResNet50, ViT, CvT) and both Phase 1 (baseline)
and Phase 2 (tuned) experiments by specifying the appropriate config YAML.

Usage:
    # Phase 1: Baseline (same config, different models)
    python scripts/train.py --model rvt --config configs/baseline.yaml
    python scripts/train.py --model cnn --config configs/baseline.yaml
    python scripts/train.py --model resnet50 --config configs/baseline.yaml
    python scripts/train.py --model vit --config configs/baseline.yaml
    python scripts/train.py --model cvt --config configs/baseline.yaml

    # Phase 2: Tuned (model-specific configs)
    python scripts/train.py --model rvt --config configs/tuned_rvt.yaml
    python scripts/train.py --model cvt --config configs/tuned_cvt.yaml
    python scripts/train.py --model vit --config configs/tuned_vit.yaml
    python scripts/train.py --model resnet50 --config configs/tuned_resnet.yaml
    python scripts/train.py --model cnn --config configs/tuned_cnn.yaml
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model, HUGGINGFACE_MODELS
from utils.dataset import build_dataframes, create_dataloaders
from utils.transforms import get_train_transform, get_val_test_transform
from utils.metrics import evaluate_model, plot_training_curves, plot_confusion_matrix, plot_roc_curve


def build_optimizer(model, cfg):
    """Build optimizer from config."""
    opt_name = cfg["optimizer"].lower()
    lr = cfg["lr"]
    wd = cfg.get("weight_decay", 0)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        momentum = cfg.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg):
    """Build LR scheduler from config (returns None if not specified)."""
    sched_name = cfg.get("scheduler")
    if sched_name is None or sched_name == "null":
        return None

    if sched_name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("step_size", 10), gamma=cfg.get("gamma", 0.5)
        )
    elif sched_name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("epochs", 100), eta_min=0
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_logits=False):
    """Run one training epoch."""
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
    """Run one validation epoch."""
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


def main():
    parser = argparse.ArgumentParser(description="Train a model for image classification")
    parser.add_argument("--model", type=str, required=True, choices=["rvt", "cnn", "resnet50", "vit", "cvt"])
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save outputs")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/figures", exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    data_cfg = cfg["data"]
    train_df, val_df, test_df = build_dataframes(
        data_cfg["positive_dir"], data_cfg["negative_dir"],
        test_size=data_cfg["test_size"], val_fraction=data_cfg.get("val_fraction", 0.176),
        random_state=data_cfg["random_state"],
    )

    train_transform = get_train_transform(data_cfg["image_size"])
    val_test_transform = get_val_test_transform(data_cfg["image_size"])
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, train_transform, val_test_transform,
        batch_size=data_cfg["batch_size"], num_workers=data_cfg.get("num_workers", 2),
    )

    # Model
    model = build_model(args.model, cfg["model"]).to(device)
    use_logits = args.model in HUGGINGFACE_MODELS
    print(f"\nModel: {args.model} | HuggingFace logits: {use_logits}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}\n")

    # Optimizer, scheduler, loss
    train_cfg = cfg["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    # Training loop
    epochs = train_cfg["epochs"]
    patience = train_cfg["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = f"{args.save_dir}/{args.model}_best.pth"

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_logits)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_logits)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print("Training complete.\n")

    # Load best model and evaluate
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    print("=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    results = evaluate_model(model, test_loader, criterion, device, use_logits)

    # Save plots
    prefix = f"{args.model}_"
    fig_dir = f"{args.save_dir}/figures"
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir=fig_dir, prefix=prefix)
    plot_confusion_matrix(results["y_true"], results["y_pred"], save_path=f"{fig_dir}/{prefix}confusion_matrix.png")
    plot_roc_curve(results["y_true"], results["y_prob"], save_path=f"{fig_dir}/{prefix}roc_curve.png")

    print(f"\nResults saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
