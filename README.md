# Automated Concrete Bridge Deck Inspection Using Unmanned Aerial System–Collected Data: A Deep Learning Approach

This repository contains the code for our comparative analysis of Residual Vision Transformer architecture against traditional CNNs for binary image classification. The study evaluates five models across three experimental phases: baseline comparison, hyperparameter optimization, and cross-validation.

## Models

| Model | Type | Description |
|-------|------|-------------|
| **RvT** | Custom | Residual Vision Transformer with inter-layer residual connections and depthwise residual projection |
| **CvT** | Pretrained | Convolutional Vision Transformer (microsoft/cvt-13) via HuggingFace |
| **ViT** | Pretrained | Vision Transformer Base/16 via timm |
| **ResNet50** | Custom | Custom ResNet-50 implementation with bottleneck blocks |
| **CNN** | Custom | Simple 2-layer CNN baseline |

## Repository Structure

```
├── models/                 # Model architectures
│   ├── __init__.py         # Model registry and factory function
│   ├── rvt.py              # Residual Vision Transformer
│   ├── cvt.py              # CvT wrapper (HuggingFace)
│   ├── vit.py              # ViT wrapper (timm)
│   ├── resnet.py           # ResNet-50
│   └── cnn.py              # Baseline CNN
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── dataset.py          # Dataset class, data splits, dataloaders
│   ├── transforms.py       # Image augmentation and preprocessing
│   └── metrics.py          # Evaluation, plotting, confusion matrix, ROC
├── configs/                # Experiment configurations (YAML)
│   ├── baseline.yaml       # Phase 1: same hyperparams for all models
│   ├── tuned_rvt.yaml      # Phase 2: optimized RvT
│   ├── tuned_cvt.yaml      # Phase 2: optimized CvT
│   ├── tuned_vit.yaml      # Phase 2: optimized ViT
│   ├── tuned_resnet.yaml   # Phase 2: optimized ResNet50
│   ├── tuned_cnn.yaml      # Phase 2: optimized CNN
│   └── kfold_rvt.yaml      # Phase 3: 5-fold cross-validation on RvT
├── scripts/                # Training scripts
│   ├── train.py            # Standard train/val/test pipeline
│   └── train_kfold.py      # K-fold cross-validation pipeline
├── notebook/
│   └── ablation/
│       ├──RvT-(Adding Residual Projection).ipynb
│       ├──RvT-(No Residual Connections or Projection.ipynb
│       └──RvT-(Only Depthwise Projection).ipynb
├── results/                # Output directory for figures
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/Kalyan0701/khatry2026automated
cd khatry2026automated
pip install -r requirements.txt
```

### Dataset

Place your dataset in the following structure:

```
Datasets/
├── Positive/    # Positive class images (.jpg)
└── Negative/    # Negative class images (.jpg)
```

## Experiments

### Phase 1: Baseline (Standardized Hyperparameters)

All models trained with AdamW (lr=1e-4, weight_decay=1e-4) to isolate architectural differences.

```bash
python scripts/train.py --model rvt      --config configs/baseline.yaml
python scripts/train.py --model vit      --config configs/baseline.yaml
python scripts/train.py --model cvt      --config configs/baseline.yaml
python scripts/train.py --model resnet50 --config configs/baseline.yaml
python scripts/train.py --model cnn      --config configs/baseline.yaml
```

### Phase 2: Optimized (Model-Specific Hyperparameters)

Each model trained with individually tuned optimizer, learning rate, and scheduler.

```bash
python scripts/train.py --model rvt      --config configs/tuned_rvt.yaml
python scripts/train.py --model vit      --config configs/tuned_vit.yaml
python scripts/train.py --model cvt      --config configs/tuned_cvt.yaml
python scripts/train.py --model resnet50 --config configs/tuned_resnet.yaml
python scripts/train.py --model cnn      --config configs/tuned_cnn.yaml
```

### Phase 3: Stratified 5-Fold Cross-Validation

Rigorous evaluation of the tuned RvT model with fresh weight initialization per fold.

```bash
python scripts/train_kfold.py --model rvt --config configs/kfold_rvt.yaml
```

## Key Results

| Model | Phase 1 (Baseline) | Phase 2 (Tuned) |
|-------|-------------------|-----------------|
| RvT | 94.05% | **96.43%** |
| ViT | 94.05% | 94.64% |
| CvT | 91.67% | 94.05% |
| ResNet50 | 92.86% | 93.45% |
| CNN | 85.12% | 87.50% |

**Phase 3 (5-Fold CV on Tuned RvT):** 95.96% ± 0.76%

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{khatry2026automated,
  title={Automated Concrete Bridge Deck Inspection Using Unmanned Aerial System--Collected Data: A Deep Learning Approach},
  author={Khatry, Kalyan and Elmi, Sayda and Samsami, Reihaneh},
  journal={ASCE OPEN: Multidisciplinary Journal of Civil Engineering},
  volume={4},
  number={1},
  pages={04026004},
  year={2026},
  publisher={American Society of Civil Engineers}
}
```
## Publication

Samsami, R., Elmi, A., & Khatry, S. (2026). *Automated Concrete Bridge Deck Inspection Using Uncrewed Aerial System (UAS)-Collected Data: A Deep Learning Approach*. ASCE Open.

📄 [Read the full paper](https://ascelibrary.org/doi/full/10.1061/AOMJAH.AOENG-0091)

## Authors

- [Kalyan Khatry](https://github.com/Kalyan0701) 
- Sayda Elmi
- [Reihaneh Samsami](https://github.com/R-SAMSAMI)

