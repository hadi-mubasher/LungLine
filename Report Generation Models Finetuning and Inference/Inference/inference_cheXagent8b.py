"""
Description
-----------
This script runs CheXagent-8B inference on a subset of the IU OpenI chest
X-ray dataset. It loads PNG images and their corresponding cleaned text
reports, constructs a test subset, and performs batched report generation
with the CheXagent-8B model using the multimodal processor interface.

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
- evaluation_csvs/IU Dataset/chexagent_8b_preds_test.csv
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
  internally for CheXagent.
- Batching is delegated to the CheXagent multimodal processor
  (text + image → input_ids, attention_mask, pixel_values).
"""

import os
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from tqdm.auto import tqdm


# ===============================================================
# CONFIGURATION
# ===============================================================
# Base IU OpenI paths and metadata CSV
DATA_CSV = "IU Dataset Download/NLMCXR_OpenI/reports_metadata.csv"
REPORT_DIR = "IU Dataset Download/NLMCXR_OpenI/reports"

# Output CSV path for CheXagent-8B predictions
OUT_CSV = "evaluation_csvs/IU Dataset/chexagent_8b_preds_test.csv"

# CheXagent-8B model identifier
MODEL_NAME = "StanfordAIMI/CheXagent-8b"

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
        - debug: if True prints file paths during loading.
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
# MODEL SETUP: Load CheXagent-8B Model and Processor
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CheXagent processor (multimodal)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Optional: left padding for decoder-only models
processor.tokenizer.padding_side = "left"

# Load CheXagent-8B model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float32,
).eval()

# Ensure pad token exists
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = model.to(device)


# ===============================================================
# SINGLE-IMAGE CHEXAGENT GENERATOR (FREE-FORM)
# ===============================================================
def _postprocess_freeform(text: str) -> str:
    """
    Lightweight cleanup for CheXagent free-form generations.

    Behavior
    --------
    - Strip SRRG-like bracketed tags if they still appear.
    - Remove Markdown bold/italics and stray asterisks.
    - Collapse multiple spaces.
    """
    if not text:
        return text

    # Remove [Anything] or [Anything: Anything Else]
    text = re.sub(r"\[[^\]]+\]\s*", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Strip Markdown bold/italics
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Remove leftover single asterisks
    text = text.replace("*", "")

    return text


def generate_report_chexagent(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    pil_L_image: Image.Image,
    view_prompt: str,
    max_new_tokens: int = 200,
    do_sample: bool = False,
) -> str:
    """
    Generate findings text for a single chest X-ray using CheXagent-8B.

    Inputs
    ------
    model : AutoModelForCausalLM
        Loaded CheXagent-8B model.
    processor : AutoProcessor
        Corresponding CheXagent processor.
    pil_L_image : PIL.Image
        Input chest X-ray in grayscale (mode 'L').
    view_prompt : str
        View token string, e.g., "[VIEW=PA]".
    max_new_tokens : int
        Maximum generation length.
    do_sample : bool
        Whether to use sampling (temperature/top-p) or beam search.

    Returns
    -------
    str
        Cleaned free-form findings text.
    """
    # Convert grayscale to RGB for the vision encoder
    rgb_img = pil_L_image.convert("RGB")

    instruction = (
        "You are a radiology assistant. Return ONLY the chest X-ray FINDINGS in 1–3 short sentences. "
        "Plain prose, no lists. DO NOT use any headings, labels, categories, or brackets. "
        "Do not prefix phrases with section names. Start directly with the findings."
    )

    # ----------------------------------------------------
    # 1. Prepare inputs (multimodal batch of size 1)
    # ----------------------------------------------------
    inputs = processor(
        text=f"{view_prompt} {instruction}",
        images=rgb_img,
        return_tensors="pt",
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    # ----------------------------------------------------
    # 2. Generate continuation
    # ----------------------------------------------------
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=(1 if do_sample else 3),
        temperature=(0.7 if do_sample else None),
        top_p=(0.9 if do_sample else None),
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    # ----------------------------------------------------
    # 3. Decode and post-process
    # ----------------------------------------------------
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    return _postprocess_freeform(response) or "[EMPTY]"


# Quick sanity check on a single sample
s0 = test_set[5]
print(
    generate_report_chexagent(
        model,
        processor,
        s0["image"],
        s0["view_prompt"],
        max_new_tokens=200,
        do_sample=False,
    )
)


# ===============================================================
# POSTPROCESSOR FOR BATCHED OUTPUT
# ===============================================================
def postprocess_freeform(text: str) -> str:
    """
    Lightweight cleanup for CheXagent free-form generations.

    Behavior
    --------
    1) Remove SRRG-style bracket tags: [Section: ...]
    2) Strip Markdown bold/italics + stray asterisks
    3) Remove CheXagent display artifacts ('>' and newlines)
    4) Remove <|box|>...</|box> and unwrap <|ref|>...</|ref>
    5) Normalize spaces + punctuation spacing

    Notes
    -----
    - Keep it deterministic and conservative (do not rewrite semantics).
    """
    if not text:
        return text

    # 1) Remove SRRG-style [Section: ...] tags
    text = re.sub(r"\[[^\]]+\]\s*", "", text)

    # 2) Strip Markdown bold/italics and stray asterisks
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = text.replace("*", "")

    # 3) Remove CheXagent display artifacts: leading '>' and newlines
    text = re.sub(r"(^|\n)>\s*", r"\1", text)
    text = text.replace("\n", " ")

    # 4) Remove <|box|>…</|box> entirely, and unwrap <|ref|>…</|ref>
    text = re.sub(r"\s*<\|box\|>.*?<\|/box\|>\s*", " ", text)
    text = re.sub(r"\s*<\|ref\|>\s*", "", text)
    text = re.sub(r"\s*<\|/ref\|>\s*", " ", text)

    # 5) Normalize spaces / punctuation
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


# ===============================================================
# BATCHED CHEXAGENT-8B INFERENCE
# ===============================================================
def batched_chexagent8b_infer(
    dataset: Dataset,
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    batch_size: int = 8,
    out_csv: str = "chexagent8b_preds.csv",
    max_new_tokens: int = 160,
    do_sample: bool = False,
    num_beams: int = 1,
) -> pd.DataFrame:
    """
    Batched CheXagent-8B inference using the multimodal processor path.

    Dataset item must be a dict with keys:
      - image (PIL; mode 'L' ok)
      - view_prompt (str)
      - report (ground-truth text, optional)
      - subject_id (optional)
      - study_id  (optional)
      - image_id / xml_file (optional, kept for downstream evaluation)

    Writes CSV with commonly used keys:
      image_id, xml_file, view, report_gt, report_pred.
    """
    model.eval()
    rows = []
    N = len(dataset)

    print(f"Batched inference (CheXagent-8B, processor): {N} samples, batch_size={batch_size}")

    # Strong anti-SRRG instruction for FINDINGS-only text
    instruction = (
        "You are a radiology assistant. Return ONLY the chest X-ray FINDINGS in 1–3 short sentences. "
        "Plain prose, no lists. DO NOT use any headings, labels, categories, or brackets. "
        "Do not prefix phrases with section names. Start directly with the findings."
    )

    # Helper: chunk indices into batches
    def chunker(it, size):
        for i in range(0, len(it), size):
            yield it[i : i + size]

    idxs = list(range(N))

    for chunk in tqdm(list(chunker(idxs, batch_size)), desc="Inferencing (batched)"):
        # ------------------------------------------------------
        # 1) Gather images and text prompts for this batch
        # ------------------------------------------------------
        metas, images, texts = [], [], []

        for i in chunk:
            s = dataset[i]
            metas.append(s)
            images.append(s["image"].convert("RGB"))  # CheXagent expects RGB
            texts.append(f"{s['view_prompt']} {instruction}")

        # ------------------------------------------------------
        # 2) Processor packs multimodal batch
        #    Produces input_ids, attention_mask, pixel_values
        # ------------------------------------------------------
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        pixel_values = inputs.get("pixel_values", None)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=(1 if do_sample else num_beams),
            temperature=(0.7 if do_sample else None),
            top_p=(0.9 if do_sample else None),
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

        # ------------------------------------------------------
        # 3) Generate batched outputs
        # ------------------------------------------------------
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if input_ids.is_cuda
                else nullcontext()
            )
            with ctx:
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    **gen_kwargs,
                )

        # ------------------------------------------------------
        # 4) Decode continuation per sample
        # ------------------------------------------------------
        for r, s in enumerate(metas):
            # Prompt length for each row (skip padded tail)
            if attention_mask is not None:
                prompt_len = attention_mask[r].sum().item()
            else:
                prompt_len = input_ids.shape[1]

            text = processor.tokenizer.decode(
                out[r, prompt_len:],
                skip_special_tokens=True,
            ).strip()

            text = postprocess_freeform(text) or "[EMPTY]"

            rows.append(
                {
                    "image_id": str(s.get("image_id", "")),
                    "xml_file": str(s.get("xml_file", "")),
                    "view": str(s.get("view_prompt", "")),
                    "report_gt": str(s.get("report", "")),
                    "report_pred": text,
                }
            )

        torch.cuda.empty_cache()

    # ------------------------------------------------------
    # 5) Save predictions to CSV
    # ------------------------------------------------------
    df_out = pd.DataFrame(rows, columns=["image_id", "xml_file", "view", "report_gt", "report_pred"])
    df_out.to_csv(out_csv, index=False)
    print(f"Saved CheXagent-8B predictions to {out_csv} (rows={len(df_out)})")
    return df_out


# ===============================================================
# MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    df_preds = batched_chexagent8b_infer(
        test_set,
        model,
        processor,
        batch_size=1,
        out_csv=OUT_CSV,
        max_new_tokens=160,
        do_sample=False,
        num_beams=3,
    )
    print(df_preds.head())
