"""
dataset.py — Phase 1/2: PyTorch Dataset for patch-level classification
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


LABEL_MAP = {"normal": 0, "tumor": 1}


def get_train_transforms(img_size=224):
    """Augmentations suited to histopathology: color jitter mimics stain variation,
    flips/rotations are safe since tissue has no canonical orientation."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(img_size=224):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class PatchDataset(Dataset):
    """
    Loads patches referenced in a manifest dataframe (as produced by tiling.py /
    utils.combine_manifests). Expects columns: slide_id, filename, label.

    Args:
        manifest_df: dataframe of patch records
        patch_root: root dir containing tumor/ and normal/ subfolders
        transforms: albumentations transform pipeline
    """

    def __init__(self, manifest_df: pd.DataFrame, patch_root: str, transforms=None):
        self.df = manifest_df.reset_index(drop=True)
        self.patch_root = Path(patch_root)
        self.transforms = transforms or get_eval_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.patch_root / row["label"] / row["filename"]
        image = np.array(Image.open(img_path).convert("RGB"))

        augmented = self.transforms(image=image)
        image_tensor = augmented["image"]

        label = LABEL_MAP[row["label"]]
        return image_tensor, torch.tensor(label, dtype=torch.long)
