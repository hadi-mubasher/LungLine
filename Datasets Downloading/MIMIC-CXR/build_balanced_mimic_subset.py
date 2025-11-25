"""
Description
-----------
This script generates balanced multi-label subsets from the
MIMIC-CXR v2.0.0 NegBio label file. It contains two sampling pipelines:

1. First Run:
   Baseline sampling attempt that balances positives and negatives per
   label without permitting label overlap. This is not valid for
   multi-label classification but is retained for reproducibility.

2. Second Run (Final Method):
   Multi-label aware balanced sampling pipeline that guarantees the
   inclusion of previously downloaded studies and performs label-wise
   balanced sampling with overlapping labels allowed.

Both outputs are written to CSV files.

Inputs
------
- mimic-cxr-2.0.0-negbio.csv
- studies-to-get-merged-with-path.csv

Outputs
-------
- mimic_cxr_balanced_subset.csv
- mimic_cxr_balanced_subset_v2.csv

Notes
-----
- Uncertain labels (-1) are removed.
- Sampling uses deterministic seeds.
- Pos_weight vector is computed for PyTorch BCEWithLogits.
"""

import pandas as pd
import numpy as np


# ======================================================================
# Helper: extract label columns by excluding metadata fields
# ======================================================================
def get_label_columns(df: pd.DataFrame) -> list:
    """
    Returns label columns for MIMIC-CXR NegBio.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    list
        List of label column names.
    """
    meta_cols = ["subject_id", "study_id", "dicom_id", "split"]
    return [c for c in df.columns if c not in meta_cols]


# ======================================================================
# FIRST RUN — baseline attempt (no label overlap)
# ======================================================================
def build_first_run_subset():
    """
    Baseline balanced sampling attempt that does not allow label overlap.
    Preserved exactly for documentation and reproducibility.

    Output
    ------
    mimic_cxr_balanced_subset.csv
    """

    # Load dataset
    df = pd.read_csv("mimic-cxr-2.0.0-negbio.csv")
    label_cols = get_label_columns(df)

    # Remove rows containing -1 across any label
    df = df[(df[label_cols] != -1).all(axis=1)].copy()
    print(f"Certain-only pool: {len(df)} rows")

    # Target count and RNG
    TARGET_TOTAL = 17500
    rng = np.random.default_rng(42)
    used_idx = set()
    selected_idx = set()

    # Compute per-label quota (pos/neg)
    n_labels = len(label_cols)
    per_label_target = TARGET_TOTAL // (2 * n_labels)
    print(f"Sampling target: ~{per_label_target} positive and negative per label")

    # Sampling helper: prevents reusing same rows
    def sample_without_replacement(condition: pd.Series, n: int):
        available = df.index[condition & (~df.index.isin(used_idx))]
        if len(available) == 0:
            return []
        n = min(len(available), n)
        chosen = rng.choice(available, n, replace=False)
        used_idx.update(chosen)
        selected_idx.update(chosen)
        return chosen

    # Compute available positive and negative counts
    stats = pd.DataFrame({
        "Pos": (df[label_cols] == 1).sum(),
        "Neg": (df[label_cols] == 0).sum()
    })
    stats["MinAvail"] = stats[["Pos", "Neg"]].min(axis=1)

    # Rare-first sampling strategy
    for label in stats.sort_values("MinAvail").index:
        pos_idx = sample_without_replacement(df[label] == 1, per_label_target)
        neg_idx = sample_without_replacement(df[label] == 0, per_label_target)
        print(f"{label:25s} -> +{len(pos_idx):4d}  -{len(neg_idx):4d}")

    # Subset assembly
    subset = df.loc[sorted(selected_idx)].copy()

    # Trim if exceeding target
    if len(subset) > TARGET_TOTAL:
        subset = subset.sample(TARGET_TOTAL, random_state=42)

    print(f"Final subset size: {len(subset)}")

    # Compute per-label balance summary
    summary = pd.DataFrame({
        "Positive_Count": (subset[label_cols] == 1).sum(),
        "Negative_Count": (subset[label_cols] == 0).sum()
    })
    summary["Positive_Ratio"] = (
        summary["Positive_Count"] /
        (summary["Positive_Count"] + summary["Negative_Count"])
    ).round(3)
    print("Label balance summary:")
    print(summary.sort_values("Positive_Ratio"))

    # Save first-run subset
    subset.to_csv("mimic_cxr_balanced_subset.csv", index=False)
    print("Saved: mimic_cxr_balanced_subset.csv")


# ======================================================================
# SECOND RUN — final multi-label aware subset builder
# ======================================================================
def build_second_run_subset():
    """
    Multi-label aware balanced sampling pipeline.

    Ensures inclusion of downloaded studies, performs label-wise balanced
    sampling with overlap permitted, trims to target size if needed, and
    computes final label statistics and pos_weight vector.

    Output
    ------
    mimic_cxr_balanced_subset_v2.csv
    """

    # --------------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------------
    df = pd.read_csv("mimic-cxr-2.0.0-negbio.csv")
    download_info = pd.read_csv("studies-to-get-merged-with-path.csv")

    label_cols = get_label_columns(df)

    # Keep rows with certain labels only
    df = df[(df[label_cols] != -1).all(axis=1)].copy()
    print(f"Certain-only pool: {len(df):,} rows")

    # Composite key to map subject+study
    df["pair_id"] = df["subject_id"].astype(str) + "_" + df["study_id"].astype(str)

    # --------------------------------------------------------------
    # Identify downloaded studies for mandatory inclusion
    # --------------------------------------------------------------
    if {"subject_id", "study_id", "downloaded"}.issubset(download_info.columns):
        download_info["pair_id"] = (
            download_info["subject_id"].astype(str) + "_" +
            download_info["study_id"].astype(str)
        )
        downloaded_subset = df[df["pair_id"].isin(
            download_info.loc[download_info["downloaded"] == 1, "pair_id"]
        )]
        print(f"Downloaded studies found: {len(downloaded_subset):,}")
    else:
        downloaded_subset = pd.DataFrame(columns=df.columns)
        print("Downloaded study information unavailable; skipping inclusion step")

    # --------------------------------------------------------------
    # Sampling configuration
    # --------------------------------------------------------------
    TARGET_FINAL = 17500
    remaining_target = max(0, TARGET_FINAL - len(downloaded_subset))

    n_labels = len(label_cols)
    per_label_target = remaining_target // (2 * n_labels)

    rng = np.random.default_rng(42)
    selected_idx = set(downloaded_subset.index)

    # --------------------------------------------------------------
    # Balanced sampling with label overlap permitted
    # --------------------------------------------------------------
    def sample_indices(condition: pd.Series, n: int) -> np.ndarray:
        available = df.index[condition]
        if len(available) == 0:
            return np.array([], dtype=int)
        n = min(len(available), n)
        return rng.choice(available, size=n, replace=False)

    for label in label_cols:
        pos_idx = sample_indices(df[label] == 1, per_label_target)
        neg_idx = sample_indices(df[label] == 0, per_label_target)
        selected_idx.update(pos_idx)
        selected_idx.update(neg_idx)
        print(f"{label:25s} -> +{len(pos_idx):4d}  -{len(neg_idx):4d}")

    subset = df.loc[list(selected_idx)].drop_duplicates()
    print(f"Subset size before trimming: {len(subset):,}")

    # --------------------------------------------------------------
    # Final trimming to target size
    # --------------------------------------------------------------
    if len(subset) > TARGET_FINAL:
        downloaded_idx = downloaded_subset.index
        non_downloaded = subset.drop(index=downloaded_idx, errors="ignore")

        needed = max(0, TARGET_FINAL - len(downloaded_idx))

        subset = pd.concat([
            downloaded_subset,
            non_downloaded.sample(
                n=min(needed, len(non_downloaded)), random_state=42
            )
        ], ignore_index=True)

        print(f"Trimmed subset size: {len(subset):,}")
    else:
        print(f"Final subset size: {len(subset):,}")

    # --------------------------------------------------------------
    # Compute per-label summary and pos_weight
    # --------------------------------------------------------------
    summary = pd.DataFrame({
        "Positive_Count": (subset[label_cols] == 1).sum(),
        "Negative_Count": (subset[label_cols] == 0).sum()
    })
    summary["Positive_Ratio"] = (
        summary["Positive_Count"] /
        (summary["Positive_Count"] + summary["Negative_Count"])
    ).round(3)
    summary["pos_weight"] = (
        summary["Negative_Count"] / summary["Positive_Count"]
    ).replace([np.inf, -np.inf], 0).fillna(0).round(3)

    print("Label balance summary:")
    print(summary.sort_values("Positive_Ratio"))

    print("pos_weight vector:")
    print(summary["pos_weight"].to_list())

    # --------------------------------------------------------------
    # Write final output
    # --------------------------------------------------------------
    subset.to_csv("mimic_cxr_balanced_subset_v2.csv", index=False)
    print("Saved: mimic_cxr_balanced_subset_v2.csv")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    build_first_run_subset()
    build_second_run_subset()
