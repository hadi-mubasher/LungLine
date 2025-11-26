"""
Description
-----------
This script runs MAIRA-2 inference on a subset of the IU OpenI chest
X-ray dataset. It loads PNG images and their corresponding cleaned text
reports, constructs a test subset, and performs batched report generation
with the MAIRA-2 model.

Inputs
------
- reports_metadata.csv (OpenI metadata)
  Expected columns include:
    - image_id
    - image_path
    - txt_file
    - empty_findings
    - report_exists
    - view

- PNG image files referenced by image_path
- Cleaned TXT reports stored under REPORT_DIR / <txt_file>

Outputs
-------
- evaluation_csvs/IU Dataset/maira2_preds_test.csv
  Columns:
    - image_id
    - xml_file
    - view
    - report_gt
    - report_pred

Notes
-----
- The script creates a test-set of 2,500 samples (seed=42) from reports
  with non-empty findings and report_exists == 1.
- Images are loaded in grayscale (PIL mode "L") and converted to RGB
  internally for MAIRA-2.
- Batching is implemented manually; MAIRA-2 inputs are padded to a
  common sequence length for each batch.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm.auto import tqdm


# ===============================================================
# CONFIGURATION
# ===============================================================
# Base IU OpenI paths and metadata CSV
DATA_CSV = "IU Dataset Download/NLMCXR_OpenI/reports_metadata.csv"
REPORT_DIR = "IU Dataset Download/NLMCXR_OpenI/reports"

# Output CSV path for MAIRA-2 predictions
OUT_CSV = "evaluation_csvs/IU Dataset/maira2_preds_test.csv"

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
        "xml_file"   : str
    }

    Notes
    -----
    - No transforms applied here; the model processor handles them.
    - PNGs are loaded in grayscale (mode="L").
    """

    def __init__(self, df: pd.DataFrame, debug: bool = False):
        """
        Input
        -----
        - df: DataFrame with at least:
              image_path, txt_file_path, view_prompt, image_id, txt_file, xml_file
        - debug: if True, prints file paths during loading.
        """
        self.df = df.reset_index(drop=True)
        self.debug = debug

    # ----------------------------------------------------------
    # Load PNG → PIL.Grayscale
    # ----------------------------------------------------------
    def load_png(self, path: str) -> Image.Image:
        """
        Loads a PNG image and returns a grayscale PIL image.
        """
        img = Image.open(path).convert("L")
        return img

    # ----------------------------------------------------------
    # Report loader
    # ----------------------------------------------------------
    def load_report(self, path: str) -> str:
        """Reads TXT report from disk."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ----------------------------------------------------------
    # __getitem__
    # ----------------------------------------------------------
    def __getitem__(self, idx: int):
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
# LOAD METADATA, FILTER, AND SAMPLE SPLITS
# ===============================================================
# Load OpenI metadata CSV
df = pd.read_csv(DATA_CSV)

# Keep only rows with actual findings and existing reports
df = df[(df["empty_findings"] == 0) & (df["report_exists"] == 1)].reset_index(drop=True)

# Add full TXT file path
df["txt_file_path"] = df["txt_file"].apply(
    lambda x: os.path.join(REPORT_DIR, x) if isinstance(x, str) else None
)

# Add view prompt in the form [VIEW=...]
df["view_prompt"] = df["view"].apply(
    lambda v: f"[VIEW={v}]" if isinstance(v, str) else "[VIEW=UNKNOWN]"
)

# Randomly select 2,500 samples for test-set
test_df = df.sample(n=2500, random_state=42).reset_index(drop=True)

# Remaining rows can be used for train/val (if needed)
remaining = df.drop(test_df.index).reset_index(drop=True)
train_df, val_df = train_test_split(remaining, test_size=0.1, random_state=42)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Build dataset for the test subset
test_set = CXRReportDataset(test_df)
print("Datasets ready.")


# ===============================================================
# MODEL SETUP: Load MAIRA-2 Model and Processor
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MAIRA-2 with remote code enabled
model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)

model = model.to(device)
model = model.eval()


# ===============================================================
# SINGLE-IMAGE MAIRA-2 FINDINGS GENERATOR
# ===============================================================
def generate_report_maira2(pil_L_image, view_prompt=None, max_new_tokens: int = 192) -> str:
    """
    Generate findings text for a single frontal chest X-ray using MAIRA-2.

    Inputs
    ------
    pil_L_image : PIL.Image
        Input chest X-ray in grayscale or RGB (converted to RGB internally).
    view_prompt : str or None
        Unused; MAIRA-2 does not rely on view tokens.
    max_new_tokens : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Generated findings text.
    """
    model.eval()

    # MAIRA-2 expects an RGB image
    pil_rgb = pil_L_image.convert("RGB")

    instruction = (
        "You are a CXR analyst. Return only chest X-ray FINDINGS in 1–3 short sentences. "
        "No headings, no patient/date/indication, no “preliminary”, no disclaimers. "
        "Do not assume sex/age/view beyond provided text. Use precise radiology wording. "
        "If abnormal, state the key positives only, concise (e.g., “Feeding tube projects to upper abdomen; "
        "bibasilar atelectasis; no pneumothorax or pleural effusion.”). "
        "Start directly with the findings."
    )

    # Build MAIRA-2 input using official helper
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=pil_rgb,
        current_lateral=None,
        prior_frontal=None,
        prior_lateral=None,
        prior_report=None,
        indication=instruction,
        technique=None,
        comparison=None,
        get_grounding=False,
        return_tensors="pt",
    )

    processed_inputs = processed_inputs.to(model.device)

    # Run generation
    with torch.no_grad():
        output_tokens = model.generate(
            **processed_inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    # Strip off the prompt portion
    prompt_len = processed_inputs["input_ids"].shape[-1]
    decoded = processor.decode(output_tokens[0][prompt_len:], skip_special_tokens=True)
    decoded = decoded.lstrip()

    # Convert MAIRA-2 structured output to plain text
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
    return prediction


# ===============================================================
# BATCHED MAIRA-2 INFERENCE
# ===============================================================
def batched_maira2_infer(
    dataset: Dataset,
    batch_size: int = 4,
    out_csv: str = OUT_CSV,
    max_new_tokens: int = 192,
) -> pd.DataFrame:
    """
    Batched MAIRA-2 inference over a CXRReportDataset.

    Requirements
    ------------
    - Global:
        - model  : MAIRA-2 model
        - processor : MAIRA-2 processor
    - Each dataset item must provide:
        - "image" (PIL.Image)
        - "view_prompt"
        - "report" (ground-truth findings)
        - "image_id"
        - "xml_file"

    Parameters
    ----------
    dataset : Dataset
        CXRReportDataset instance.
    batch_size : int
        Number of samples per batch.
    out_csv : str
        Path to output CSV file.
    max_new_tokens : int
        Maximum number of tokens to generate per sample.

    Returns
    -------
    pd.DataFrame
        DataFrame containing image IDs, metadata, and predictions.
    """
    model.eval()
    rows = []
    N = len(dataset)

    print(f"Batched MAIRA-2 inference: {N} samples, batch_size={batch_size}")

    instruction = (
        "You are a CXR analyst. Return only chest X-ray FINDINGS in 1–3 short sentences. "
        "No headings, no patient/date/indication, no “preliminary”, no disclaimers. "
        "Do not assume sex/age/view beyond provided text. Use precise radiology wording. "
        "If abnormal, state the key positives only, concise (e.g., “Feeding tube projects to upper abdomen; "
        "bibasilar atelectasis; no pneumothorax or pleural effusion.”). "
        "Start directly with the findings."
    )

    # Helper to split indices into chunks
    def chunker(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    idxs = list(range(N))

    # Main batch loop
    for chunk in tqdm(list(chunker(idxs, batch_size)), desc="Inferencing (MAIRA-2 batched)"):
        batch_inputs = []
        metas = []

        # -------------------------------------------------------
        # Build per-sample MAIRA-2 inputs
        # -------------------------------------------------------
        for i in chunk:
            s = dataset[i]
            pil_rgb = s["image"].convert("RGB")

            processed = processor.format_and_preprocess_reporting_input(
                current_frontal=pil_rgb,
                current_lateral=None,
                prior_frontal=None,
                prior_lateral=None,
                prior_report=None,
                indication=instruction,
                technique=None,
                comparison=None,
                get_grounding=False,
                return_tensors="pt",
            )

            processed = {k: v.to(model.device) for k, v in processed.items()}
            batch_inputs.append(processed)
            metas.append(s)

        # -------------------------------------------------------
        # Manual collation with padding
        # -------------------------------------------------------
        max_len = max(p["input_ids"].shape[1] for p in batch_inputs)

        input_ids = []
        attn_masks = []
        images = []

        for proc in batch_inputs:
            pad_len = max_len - proc["input_ids"].shape[1]

            ids = torch.cat(
                [
                    proc["input_ids"],
                    torch.full(
                        (1, pad_len),
                        processor.tokenizer.pad_token_id,
                        device=model.device,
                        dtype=torch.long,
                    ),
                ],
                dim=1,
            )

            mask = torch.cat(
                [
                    proc["attention_mask"],
                    torch.zeros((1, pad_len), device=model.device, dtype=torch.long),
                ],
                dim=1,
            )

            input_ids.append(ids)
            attn_masks.append(mask)
            images.append(proc["pixel_values"])  # already (1,3,H,W)

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attn_masks, dim=0)
        pixel_values = torch.cat(images, dim=0)  # (B,3,H,W)

        # -------------------------------------------------------
        # Run MAIRA-2 generation on the batch
        # -------------------------------------------------------
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        # -------------------------------------------------------
        # Decode predictions per sample
        # -------------------------------------------------------
        for b_idx, meta in enumerate(metas):
            prompt_len = attention_mask[b_idx].sum().item()
            pred_tokens = out[b_idx, prompt_len:]

            decoded = processor.decode(pred_tokens, skip_special_tokens=True).lstrip()
            plain = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)

            rows.append(
                {
                    "image_id": str(meta.get("image_id", "")),
                    "xml_file": str(meta.get("xml_file", "")),
                    "view": str(meta.get("view_prompt", "")),
                    "report_gt": str(meta.get("report", "")),
                    "report_pred": plain,
                }
            )

        # Free batch tensors
        del out, input_ids, attention_mask, pixel_values
        torch.cuda.empty_cache()

    # Save results to CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved MAIRA-2 predictions to {out_csv} (rows={len(df_out)})")
    return df_out


# ===============================================================
# MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    batched_maira2_infer(test_set, out_csv=OUT_CSV)
