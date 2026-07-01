"""
utils.py — shared helpers for the histopathology pipeline
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def patient_level_split(manifest_df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42):
    """
    Splits a patch manifest into train/val/test by PATIENT (slide_id), not by patch.

    This is critical: patches from the same slide are highly correlated (same
    staining, same tissue characteristics). Splitting by patch instead of patient
    leaks information and gives falsely optimistic metrics — a very common
    beginner mistake in WSI classification.

    Args:
        manifest_df: dataframe with at least a 'slide_id' column
        val_size, test_size: fractions of PATIENTS (not patches) to hold out
        seed: random seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    patient_ids = manifest_df["slide_id"].unique()

    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=(val_size + test_size), random_state=seed
    )
    relative_test_size = test_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=relative_test_size, random_state=seed
    )

    train_df = manifest_df[manifest_df["slide_id"].isin(train_ids)].reset_index(drop=True)
    val_df = manifest_df[manifest_df["slide_id"].isin(val_ids)].reset_index(drop=True)
    test_df = manifest_df[manifest_df["slide_id"].isin(test_ids)].reset_index(drop=True)

    print(
        f"Patients -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}\n"
        f"Patches  -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def combine_manifests(manifest_dir: str) -> pd.DataFrame:
    """Combines all per-slide manifest CSVs (produced by tiling.py) into one dataframe."""
    import glob
    import os

    files = glob.glob(os.path.join(manifest_dir, "*_manifest.csv"))
    if not files:
        raise FileNotFoundError(f"No manifest CSVs found in {manifest_dir}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)
