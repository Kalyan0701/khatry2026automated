"""
Evaluation metrics and visualization utilities.

Provides functions for:
  - Model evaluation on a test set
  - Plotting training/validation loss and accuracy curves
  - Confusion matrix heatmap
  - ROC curve with AUC score
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, test_loader, criterion, device, use_logits=False):
    """
    Evaluate a model on the test set.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        criterion: Loss function.
        device: torch.device.
        use_logits: If True, access outputs via `.logits` (for HuggingFace models).

    Returns:
        Dict with keys: test_loss, test_acc, y_true, y_pred, y_prob.
    """
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            logits = outputs.logits if use_logits else outputs
            loss = criterion(logits, labels)
            test_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    test_loss /= total
    test_acc = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["NEGATIVE", "POSITIVE"]))

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir=None, prefix=""):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        train_accs: List of training accuracies per epoch.
        val_accs: List of validation accuracies per epoch.
        save_dir: Optional directory to save figures.
        prefix: Optional prefix for filenames.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", color="blue", linewidth=2, marker="o", markersize=4)
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange", linewidth=2, marker="s", markersize=4)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{prefix}loss_curves.png", dpi=150)
    plt.show()

    # Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label="Training Accuracy", color="green", linewidth=2, marker="o", markersize=4)
    plt.plot(epochs, val_accs, label="Validation Accuracy", color="red", linewidth=2, marker="s", markersize=4)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/{prefix}accuracy_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NEGATIVE", "POSITIVE"],
                yticklabels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot the ROC curve with AUC score."""
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"AUC Score: {auc:.4f}")
    return auc
