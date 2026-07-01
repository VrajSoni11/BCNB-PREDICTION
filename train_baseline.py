"""
train_baseline.py — Phase 2: Baseline patch-level tumor classifier

Fine-tunes a pretrained CNN (default: EfficientNet-B0 via timm) on tissue patches.
Built with free-tier GPU constraints in mind:
  - mixed precision (AMP) to cut memory usage ~40-50%
  - modest batch size with optional gradient accumulation
  - checkpointing every epoch (Kaggle/Colab sessions can disconnect — don't lose progress)

Usage:
    python train_baseline.py --patch_root patches/ --manifest_dir patches/ \
                              --epochs 10 --batch_size 32 --out_dir checkpoints/
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

from dataset import PatchDataset, get_train_transforms, get_eval_transforms
from utils import combine_manifests, patient_level_split


def build_model(model_name="efficientnet_b0", pretrained=True, num_classes=2):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "eval"):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    return {"loss": avg_loss, "auroc": auroc, "precision": precision, "recall": recall, "f1": f1}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print(
            "WARNING: no GPU detected. This script will be extremely slow on CPU. "
            "Make sure GPU accelerator is enabled (Kaggle: Settings > Accelerator > GPU)."
        )

    # --- Data ---
    manifest = combine_manifests(args.manifest_dir)
    train_df, val_df, test_df = patient_level_split(manifest)

    train_ds = PatchDataset(train_df, args.patch_root, transforms=get_train_transforms())
    val_ds = PatchDataset(val_df, args.patch_root, transforms=get_eval_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- Model ---
    model = build_model(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()

        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        # checkpoint every epoch — sessions can disconnect, don't lose progress
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "val_metrics": val_metrics},
            out_dir / "last_checkpoint.pt",
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"New best AUROC: {best_auroc:.4f} — saved best_model.pt")

    print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_root", required=True, help="dir containing tumor/ and normal/ subfolders")
    parser.add_argument("--manifest_dir", required=True, help="dir containing *_manifest.csv files")
    parser.add_argument("--out_dir", default="checkpoints/")
    parser.add_argument("--model_name", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32, help="reduce to 16 if you hit OOM on free GPU")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    main(args)
