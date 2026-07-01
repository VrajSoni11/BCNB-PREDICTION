"""
tiling.py — Phase 1: Whole-Slide Image tiling pipeline

Slices gigapixel WSIs into fixed-size patches, discards background/non-tissue
patches, and maps tumor annotation contours to patch-level labels.

Usage:
    python tiling.py --wsi_path slide.svs --annotation_path slide_annot.json \
                      --out_dir patches/ --patch_size 512 --level 1

Requires: openslide-python, opencv-python, numpy, Pillow
On Kaggle/Colab: apt-get install -y openslide-tools  (before pip install openslide-python)
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None  # allows --help / dry runs without the system lib installed


def is_tissue_patch(patch: np.ndarray, tissue_threshold: float = 0.1) -> bool:
    """
    Simple Otsu-threshold based tissue detector.
    Returns True if enough of the patch is tissue (not background/whitespace).

    This is intentionally simple — good enough for Phase 1. Swap for a trained
    tissue segmenter later if you want cleaner filtering.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tissue_ratio = np.count_nonzero(thresh) / thresh.size
    return tissue_ratio > tissue_threshold


def load_tumor_mask(annotation_path: str, slide_dims: tuple) -> np.ndarray:
    """
    Loads tumor region contours (expects a JSON list of polygon point lists,
    matching BCNB's annotation format) and rasterizes them into a binary mask
    at full slide resolution (level 0 coordinates).

    Adjust this parser to match the exact BCNB annotation schema once you have
    real files in hand — this is a reasonable starting assumption.
    """
    mask = np.zeros(slide_dims[::-1], dtype=np.uint8)  # (height, width)
    if not annotation_path or not os.path.exists(annotation_path):
        return mask

    with open(annotation_path, "r") as f:
        contours_raw = json.load(f)

    for contour in contours_raw:
        pts = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=1)

    return mask


def tile_slide(
    wsi_path: str,
    out_dir: str,
    annotation_path: str = None,
    patch_size: int = 512,
    level: int = 1,
    stride: int = None,
    tissue_threshold: float = 0.1,
    tumor_overlap_threshold: float = 0.2,
):
    """
    Core tiling function.

    Args:
        wsi_path: path to the .svs / .tiff whole-slide image
        out_dir: where to save extracted patches
        annotation_path: path to tumor contour annotation (optional)
        patch_size: patch width/height in pixels (at chosen level)
        level: WSI pyramid level to read from (0 = full res, higher = downsampled)
        stride: step size between patches; defaults to patch_size (non-overlapping)
        tissue_threshold: min fraction of patch that must be tissue to keep it
        tumor_overlap_threshold: min fraction of patch overlapping tumor mask
                                  to label the patch as "tumor"

    Outputs:
        patches saved as PNGs in out_dir/tumor/ and out_dir/normal/
        a manifest.csv listing all saved patches + labels
    """
    if openslide is None:
        raise RuntimeError(
            "openslide not installed. On Kaggle/Colab run: "
            "!apt-get install -y openslide-tools && pip install openslide-python"
        )

    stride = stride or patch_size
    slide = openslide.OpenSlide(wsi_path)

    level_dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    tumor_mask = load_tumor_mask(annotation_path, slide.level_dimensions[0])

    tumor_dir = Path(out_dir) / "tumor"
    normal_dir = Path(out_dir) / "normal"
    tumor_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    slide_id = Path(wsi_path).stem

    n_x = level_dims[0] // stride
    n_y = level_dims[1] // stride

    for iy in range(n_y):
        for ix in range(n_x):
            # coordinates at level 0 (openslide always reads relative to level 0 origin)
            x0 = int(ix * stride * downsample)
            y0 = int(iy * stride * downsample)

            patch = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            if not is_tissue_patch(patch_np, tissue_threshold):
                continue

            # check tumor overlap using the full-res mask, cropped to this patch's region
            mask_crop = tumor_mask[
                y0 : y0 + int(patch_size * downsample),
                x0 : x0 + int(patch_size * downsample),
            ]
            tumor_fraction = mask_crop.mean() if mask_crop.size else 0.0
            label = "tumor" if tumor_fraction > tumor_overlap_threshold else "normal"

            save_dir = tumor_dir if label == "tumor" else normal_dir
            fname = f"{slide_id}_x{ix}_y{iy}.png"
            patch.save(save_dir / fname)

            manifest_rows.append(
                {"slide_id": slide_id, "filename": fname, "label": label, "x": x0, "y": y0}
            )

    import pandas as pd

    manifest_path = Path(out_dir) / f"{slide_id}_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved {len(manifest_rows)} patches. Manifest: {manifest_path}")

    return manifest_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile a WSI into patches for training.")
    parser.add_argument("--wsi_path", required=True)
    parser.add_argument("--annotation_path", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()

    tile_slide(
        wsi_path=args.wsi_path,
        out_dir=args.out_dir,
        annotation_path=args.annotation_path,
        patch_size=args.patch_size,
        level=args.level,
        stride=args.stride,
    )
