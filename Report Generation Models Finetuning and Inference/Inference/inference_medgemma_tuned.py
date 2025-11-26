"""
Description
-----------
This script runs a *tuned* MedGemma-4B-IT model (with LoRA adapters) on a
subset of the IU OpenI chest X-ray dataset. It loads PNG images and their
corresponding cleaned text reports, constructs a test subset, and performs
batched report generation with the MedGemma-4B-IT vision-language model.

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
    - (optional) xml_file

- PNG image files referenced by image_path
- Cleaned TXT reports stored under REPORT_DIR / <txt_file>
- LoRA-tuned MedGemma checkpoint directory:
    LORA_DIR = "cxr_checkpoints/medgemma"

Outputs
-------
- evaluation_csvs/IU Dataset/medgemma_tuned_preds_test.csv
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
  internally for MedGemma.
- Batching uses the official chat template with image tokens and pads
  text/image inputs via the model's processor.
- The base MedGemma-4B-IT weights are loaded from MODEL_NAME and then
  merged with LoRA adapters loaded from LORA_DIR.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


# ===============================================================
# CONFIGURATION
# ===============================================================
# Base IU OpenI paths and metadata CSV
DATA_CSV = "IU Dataset Download/NLMCXR_OpenI/reports_metadata.csv"
REPORT_DIR = "IU Dataset Download/NLMCXR_OpenI/reports"

# Output CSV path for tuned MedGemma predictions
OUT_CSV = "evaluation_csvs/IU Dataset/medgemma_tuned_preds_test.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# MedGemma base model name + LoRA adapter directory
MODEL_NAME = "google/medgemma-4b-it"
LORA_DIR   = "cxr_checkpoints/medgemma"   # your saved adapters + processor


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
        "xml_file"   : str (optional, may be empty)
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
              image_path, txt_file_path, view_prompt, image_id, txt_file
              (xml_file optional)
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
print("✅ Datasets ready.")


# ===============================================================
# MODEL SETUP: Load *tuned* MedGemma-4B-IT (LoRA) and Processor
# ===============================================================
# Load processor from LORA_DIR (so it matches your fine-tuning setup)
processor = AutoProcessor.from_pretrained(LORA_DIR)

# Load base MedGemma weights
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Attach LoRA adapters
model = PeftModel.from_pretrained(model, LORA_DIR)
model.eval()

tok = processor.tokenizer
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


# ===============================================================
# SINGLE-IMAGE MedGemma FINDINGS GENERATOR
# ===============================================================
def generate_report_medgemma(
    pil_L_image: Image.Image,
    view_prompt: str,
    max_new_tokens: int = 192,
    do_sample: bool = False,
) -> str:
    """
    Generate findings text for a single frontal chest X-ray using MedGemma.

    Inputs
    ------
    pil_L_image : PIL.Image
        Input chest X-ray in grayscale or RGB (converted to RGB internally).
    view_prompt : str
        View token string (e.g., "[VIEW=PA]") prepended to the instruction.
    max_new_tokens : int
        Maximum number of tokens to generate.
    do_sample : bool
        Whether to use sampling (True) or deterministic search (False).

    Returns
    -------
    str
        Generated findings text.
    """
    model.eval()
    instruction = """You are a CXR analyst. Return only chest X-ray FINDINGS in 1–3 short sentences. No headings, no patient/date/indication,
    no “preliminary”, no disclaimers. Do not assume sex/age/view beyond provided text. Use precise radiology wording. If normal,
    write a single normal line (e.g., “Lungs clear. Cardiomediastinal silhouette normal. No pleural effusion or pneumothorax.”).
    If abnormal, state the key positives only, concise (e.g., “Feeding tube projects to upper abdomen; bibasilar atelectasis; no pneumothorax or pleural effusion.”).
    Start directly with the findings."""

    # 1) Messages: include an image slot (no URL) + your text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # reserves the <image> token
                {"type": "text", "text": f"{view_prompt} {instruction}"},
            ],
        }
    ]

    # 2) Turn messages into a text prompt that contains the image token
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,  # get the raw string with the image token
    )

    # 3) Provide the actual image separately; convert to RGB
    pil_rgb = pil_L_image.convert("RGB")
    inputs = processor(
        text=prompt,
        images=[pil_rgb],  # images length must match number of image tokens in the prompt (here: 1)
        return_tensors="pt",
    ).to(model.device)

    # 4) Generate
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7 if do_sample else None,
            top_p=0.9 if do_sample else None,
            num_beams=3,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )

    # 5) Decode only the newly generated tokens
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# ===============================================================
# BATCHED MedGemma INFERENCE
# ===============================================================
def strip_role_first_line(t: str) -> str:
    """
    Removes a leading 'model/assistant/system:' line if present,
    then returns the remaining text.
    """
    first, _, rest = t.partition("\n")
    if first.strip().lower().rstrip(":") in {"model", "assistant", "system"}:
        return rest.lstrip()
    return t


def batched_medgemma_infer(
    dataset: Dataset,
    batch_size: int = 8,
    out_csv: str = OUT_CSV,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    num_beams: int = 1,
) -> pd.DataFrame:
    """
    Batched MedGemma inference over a CXRReportDataset.

    Requirements
    ------------
    - Global:
        - model      : tuned MedGemma-4B-IT model (base + LoRA)
        - processor  : MedGemma processor loaded from LORA_DIR
        - tok        : tokenizer associated with processor
    - Each dataset item must provide:
        - "image" (PIL.Image; grayscale accepted)
        - "view_prompt"
        - "report" (ground-truth findings)
        - "image_id"
        - "xml_file" (optional)

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
    do_sample : bool
        Whether to use sampling or deterministic search.
    num_beams : int
        Number of beams for beam search (if do_sample=False).

    Returns
    -------
    pd.DataFrame
        DataFrame containing image IDs, metadata, and predictions.
    """
    model.eval()
    rows: List[Dict[str, Any]] = []
    N = len(dataset)
    print(f"🚀 Batched MedGemma inference: {N} samples, batch_size={batch_size}")

    instruction = """You are a CXR analyst. Return only chest X-ray FINDINGS in 1–3 short sentences. No headings, no patient/date/indication,
    no “preliminary”, no disclaimers. Do not assume sex/age/view beyond provided text. Use precise radiology wording. If normal,
    write a single normal line (e.g., “Lungs clear. Cardiomediastinal silhouette normal. No pleural effusion or pneumothorax.”).
    If abnormal, state the key positives only, concise (e.g., “Feeding tube projects to upper abdomen; bibasilar atelectasis; no pneumothorax or pleural effusion.”).
    Start directly with the findings."""

    def chunker(it, size):
        for i in range(0, len(it), size):
            yield it[i : i + size]

    idxs = list(range(N))
    for chunk in tqdm(list(chunker(idxs, batch_size)), desc="Inferencing (MedGemma batched)"):
        prompts: List[str] = []
        imgs: List[Image.Image] = []
        metas: List[Dict[str, Any]] = []

        # -------------------------------------------------------
        # Build prompts + collect images for this batch
        # -------------------------------------------------------
        for i in chunk:
            s = dataset[i]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{s['view_prompt']} {instruction}"},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(prompt)
            imgs.append(s["image"].convert("RGB"))
            metas.append(s)

        # MedGemma expects images as list-of-lists when text is a list
        images_per_prompt = [[img] for img in imgs]

        inputs = processor(
            text=prompts,
            images=images_per_prompt,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=(0.7 if do_sample else None),
                top_p=(0.9 if do_sample else None),
                num_beams=num_beams,
                use_cache=True,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

        # Slice each sample's newly generated tokens using true prompt lengths
        prompt_lens = inputs["attention_mask"].sum(dim=1)  # (batch,)
        for row_idx, s in enumerate(metas):
            gen_tokens = out[row_idx, prompt_lens[row_idx] :]
            text = processor.decode(gen_tokens, skip_special_tokens=True).strip()
            text = strip_role_first_line(text)
            rows.append(
                {
                    "image_id": str(s.get("image_id", "")),
                    "xml_file": str(s.get("xml_file", "")),
                    "view": str(s.get("view_prompt", "")),
                    "report_gt": str(s.get("report", "")),
                    "report_pred": text,
                }
            )

        del inputs, out, images_per_prompt
        torch.cuda.empty_cache()

    df_out = pd.DataFrame(rows, columns=["image_id", "xml_file", "view", "report_gt", "report_pred"])
    df_out.to_csv(out_csv, index=False)
    print(f"📁 Saved → {out_csv}  (rows={len(df_out)})")
    return df_out


# ===============================================================
# MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    batched_medgemma_infer(
        test_set,
        batch_size=8,
        out_csv=OUT_CSV,
        max_new_tokens=128,
        do_sample=False,
        num_beams=3,
    )
