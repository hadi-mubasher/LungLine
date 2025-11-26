"""
Description
-----------
This script runs CXR-LLAVA-v2 inference on a subset of the IU OpenI chest
X-ray dataset. It loads PNG images and their corresponding cleaned text
reports, constructs a test subset, and generates free-text radiology
reports using the CXR-LLAVA-v2 model.

Inputs
------
- IU Dataset Download/NLMCXR_OpenI/reports_metadata.csv
  Expected columns include:
    - image_id
    - image_path
    - txt_file
    - xml_file
    - empty_findings
    - report_exists
    - view

- PNG image files referenced by image_path
- Cleaned TXT reports stored under:
    IU Dataset Download/NLMCXR_OpenI/reports/<txt_file>

Outputs
-------
- evaluation_csvs/IU Dataset/llava_v2_predictions_test.csv
  Columns:
    - image_id
    - xml_file
    - view
    - report_gt
    - report_pred

Notes
-----
- The script creates a test set of 2,500 samples (seed=42) from rows with
  non-empty findings and report_exists == 1.
- Images are loaded in grayscale (PIL mode "L") and passed to the model
  via the CXR-LLAVA-v2 ask_question API.
- The batch loop is robust to per-sample failures: exceptions are logged
  and the loop continues, but failed samples are not written to the CSV.
"""

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from transformers import AutoModel
from tqdm.auto import tqdm


# ===============================================================
# CONFIGURATION
# ===============================================================
DATA_CSV = "IU Dataset Download/NLMCXR_OpenI/reports_metadata.csv"
REPORT_DIR = "IU Dataset Download/NLMCXR_OpenI/reports"

OUT_CSV = "evaluation_csvs/IU Dataset/llava_v2_predictions_test.csv"
MODEL_NAME = "ECOFRI/CXR-LLAVA-v2"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


# ===============================================================
# DATASET: PNG + TXT Report Loader
# ===============================================================
class CXRReportDataset(Dataset):
    """
    Dataset loader for paired (PNG → image), (TXT → report).

    Output per sample
    -----------------
    {
        "image"      : PIL.Image in grayscale "L"
        "view_prompt": str, e.g., "[VIEW=PA]"
        "report"     : str (raw cleaned report)
        "image_id"   : str
        "txt_file"   : str
        "xml_file"   : str (if present in metadata, else "")
    }

    Notes
    -----
    - No transforms are applied here; the model code handles them.
    - PNG files are loaded in grayscale (mode="L").
    """

    def __init__(self, df: pd.DataFrame, debug: bool = False):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least:
              - image_path
              - txt_file_path
              - view_prompt
              - image_id
              - txt_file
              - xml_file (optional)
        debug : bool, optional
            If True, prints file paths during loading.
        """
        self.df = df.reset_index(drop=True)
        self.debug = debug

    # ----------------------------------------------------------
    # Load PNG → PIL.Grayscale
    # ----------------------------------------------------------
    def load_png(self, path: str) -> Image.Image:
        """
        Load a PNG image from disk and return a grayscale PIL image.

        Parameters
        ----------
        path : str
            Path to the PNG file.

        Returns
        -------
        PIL.Image.Image
            Grayscale image (mode="L").
        """
        img = Image.open(path).convert("L")
        return img

    # ----------------------------------------------------------
    # Report loader
    # ----------------------------------------------------------
    def load_report(self, path: str) -> str:
        """
        Read a TXT report from disk.

        Parameters
        ----------
        path : str
            Path to the TXT file.

        Returns
        -------
        str
            Raw report text.
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ----------------------------------------------------------
    # __getitem__
    # ----------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        rpt_path = row["txt_file_path"]
        prompt = row.get("view_prompt", "[VIEW=UNKNOWN]")

        if self.debug:
            print(f"[DEBUG] IMAGE : {img_path}")
            print(f"[DEBUG] REPORT: {rpt_path}")

        image = self.load_png(img_path)
        report = self.load_report(rpt_path)

        return {
            "image": image,
            "view_prompt": prompt,
            "report": report,
            "image_id": row["image_id"],
            "txt_file": row["txt_file"],
            "xml_file": row.get("xml_file", ""),
        }

    def __len__(self) -> int:
        return len(self.df)


# ===============================================================
# LOAD METADATA, FILTER ROWS, SAMPLE SPLITS
# ===============================================================
# Load OpenI metadata CSV
df = pd.read_csv(DATA_CSV)

# Keep only rows with actual findings and an existing report file
df = df[(df["empty_findings"] == 0) & (df["report_exists"] == 1)].reset_index(drop=True)

# Add full TXT file path
df["txt_file_path"] = df["txt_file"].apply(
    lambda x: os.path.join(REPORT_DIR, x) if isinstance(x, str) else None
)

# Add view prompt in the form [VIEW=...]
df["view_prompt"] = df["view"].apply(
    lambda v: f"[VIEW={v}]" if isinstance(v, str) else "[VIEW=UNKNOWN]"
)

# Randomly select 2,500 samples for test set
test_df = df.sample(n=2500, random_state=42).reset_index(drop=True)

# Remaining rows can be used for train/val if needed
remaining = df.drop(test_df.index).reset_index(drop=True)
train_df, val_df = train_test_split(remaining, test_size=0.1, random_state=42)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Build test dataset
test_set = CXRReportDataset(test_df)
print("Datasets ready.")


# ===============================================================
# MODEL SETUP: Load CXR-LLAVA-v2
# ===============================================================
# Select device and dtype
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
dtype = torch.bfloat16 if use_cuda else torch.float32

# Load remote-code model and move to device
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=dtype,
).eval().to(device)

# Quiet chat-template warning for tokenizer if needed
tok = model.tokenizer
if getattr(tok, "chat_template", None) in (None, ""):
    tok.chat_template = "{{ bos_token }}{% for m in messages %}{{ m['content'] + '\n' }}{% endfor %}"

print(f"Model loaded on {device} with dtype={dtype}.")


# ===============================================================
# HELPER: Ensure PIL grayscale image
# ===============================================================
def to_pil_L(x: Any) -> Image.Image:
    """
    Convert input to a PIL grayscale image (mode="L").

    Parameters
    ----------
    x : Any
        PIL.Image, numpy array, or path-like.

    Returns
    -------
    PIL.Image.Image
        Grayscale image (mode="L").
    """
    if isinstance(x, Image.Image):
        return x.convert("L")

    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return Image.fromarray(x.astype(np.uint8)).convert("L")
        if x.ndim == 3 and x.shape[-1] == 3:
            return Image.fromarray(x.astype(np.uint8)).convert("L")

    # Fallback for path-like inputs
    return Image.open(x).convert("L")


# ===============================================================
# BATCH INFERENCE LOOP
# ===============================================================
def run_batch_inference(
    dataset: CXRReportDataset,
    out_csv: str = OUT_CSV,
) -> pd.DataFrame:
    """
    Run CXR-LLAVA-v2 report generation over a CXRReportDataset and write to CSV.

    Parameters
    ----------
    dataset : CXRReportDataset
        Test dataset containing image, view_prompt, report, image_id and xml_file.
    out_csv : str
        Path to the output CSV file.

    Behavior
    --------
    - Calls model.ask_question(question=..., image=...) for each sample.
    - Logs any per-sample exceptions and continues.
    - Only successful samples are written to the output CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: image_id, xml_file, view, report_gt, report_pred.
    """
    rows = []
    total = len(dataset)
    print(f"Running batch inference on {total} samples...")

    for i in tqdm(range(total), desc="Inferencing (CXR-LLAVA-v2)"):
        sample = dataset[i]
        img = to_pil_L(sample["image"])

        try:
            pred = model.ask_question(
                question="Analyze the CXR image and write the diagnosis report.",
                image=img,
            )

            rows.append(
                {
                    "image_id": str(sample.get("image_id", "")),
                    "xml_file": str(sample.get("xml_file", "")),
                    "view": str(sample.get("view_prompt", "")),
                    "report_gt": str(sample.get("report", "")),
                    "report_pred": pred,
                }
            )

        except Exception as e:
            # Log and continue; this sample is skipped in the output CSV
            print(f"[Error] index {i}: {type(e).__name__}: {e}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv} (rows={len(out_df)})")
    return out_df


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    _ = run_batch_inference(test_set, out_csv=OUT_CSV)
