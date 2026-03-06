"""
Dataset utilities for loading and splitting image data.

Handles:
  - Building a DataFrame of image paths and labels from directory structure
  - Custom PyTorch Dataset for loading images with transforms
  - Train/val/test splitting with stratification
  - DataLoader creation
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """
    PyTorch Dataset that loads images from file paths stored in a DataFrame.

    Args:
        dataframe: DataFrame with 'Filepath' and 'Label' columns.
        transform: torchvision transforms to apply to each image.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Filepath']
        label_str = self.dataframe.iloc[idx]['Label']
        label = 1 if label_str == "POSITIVE" else 0

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def generate_df(image_dir, label):
    """
    Create a DataFrame of image file paths and their corresponding labels.

    Args:
        image_dir: Path to directory containing .jpg images.
        label: String label to assign (e.g., 'POSITIVE' or 'NEGATIVE').

    Returns:
        DataFrame with 'Filepath' and 'Label' columns.
    """
    image_dir = Path(image_dir)
    filepaths = pd.Series(list(image_dir.glob("*.jpg")), name="Filepath").astype(str)
    labels = pd.Series(label, name="Label", index=filepaths.index)
    return pd.concat([filepaths, labels], axis=1)


def build_dataframes(positive_dir, negative_dir, test_size=0.20, val_fraction=0.176, random_state=42):
    """
    Build train/val/test DataFrames with stratified splits.

    Split ratios (default):
      - Test:  20% of total
      - Val:   ~15% of total (17.6% of remaining 80%)
      - Train: ~65% of total

    Args:
        positive_dir: Path to directory of positive-class images.
        negative_dir: Path to directory of negative-class images.
        test_size: Fraction of data reserved for testing.
        val_fraction: Fraction of train+val reserved for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    positive_df = generate_df(positive_dir, label="POSITIVE")
    negative_df = generate_df(negative_dir, label="NEGATIVE")
    all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

    train_val_df, test_df = train_test_split(
        all_df, test_size=test_size, stratify=all_df["Label"], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_fraction, stratify=train_val_df["Label"], random_state=random_state
    )

    print(f"Training set size:   {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size:       {len(test_df)}")

    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, train_transform, val_test_transform, batch_size=16, num_workers=2):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=val_test_transform)
    test_dataset = ImageDataset(test_df, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
