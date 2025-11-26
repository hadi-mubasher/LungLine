"""
Description
-----------
This script runs CheXagent-2-3B (tokenizer-only mode) inference on a subset
of the IU OpenI chest X-ray dataset. It loads PNG images and their
corresponding cleaned TXT reports, builds a 2,500-sample test subset, and
performs FINDINGS-only report generation using the decoder-only CheXagent
model via its tokenizer-only chat template mechanism.

Inputs
------
- reports_metadata.csv (OpenI metadata)
  Required columns:
    - image_id
    - image_path
    - txt_file
    - empty_findings
    - report_exists
    - view

- PNG image files in paths provided by `image_path`
- Cleaned TXT reports stored under REPORT_DIR / <txt_file>

Outputs
-------
- evaluation_csvs/IU Dataset/chexagent2_3b_preds_test.csv
  Columns:
    - image_id
    - xml_file
    - view
    - report_gt
    - report_pred

Notes
-----
- Uses tokenizer.from_list_format() and tokenizer.apply_chat_template()
  to prepare multimodal prompts (image + text).
- Images are saved temporarily as JPEGs because the tokenizer-only path
  expects on-disk image objects.
- All generation is FINDINGS-only with strong anti-SRRG instructions.
- Batching is fully manual because CheXagent-2-3B does not support
  multimodal batching internally.
"""

import os
import re
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ===============================================================
# CONFIGURATION
# ===============================================================
DATA_CSV    = "IU Dataset Download/NLMCXR_OpenI/reports_metadata.csv"
REPORT_DIR  = "IU Dataset Download/NLMCXR_OpenI/reports"
OUT_CSV     = "evaluation_csvs/IU Dataset/chexagent2_3b_preds_test.csv"

MODEL = "StanfordAIMI/CheXagent-2-3b"

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
        "view_prompt": str     (e.g., "[VIEW=PA]")
        "report"     : str     (ground-truth text)
        "image_id"   : str
        "txt_file"   : str
        "xml_file"   : optional
    }

    Notes
    -----
    - No transforms applied here; tokenizer/model paths handle preprocessing.
    - PNG images are loaded in grayscale.
    """

    def __init__(self, df: pd.DataFrame, debug: bool = False):
        """
        Parameters
        ----------
        df : DataFrame
            Must contain: image_path, txt_file_path, view_prompt, image_id, txt_file.
        debug : bool
            When True, prints paths during item loading.
        """
        self.df = df.reset_index(drop=True)
        self.debug = debug

    # ----------------------------------------------------------
    # Load PNG → PIL.Grayscale
    # ----------------------------------------------------------
    def load_png(self, path: str) -> Image.Image:
        img = Image.open(path).convert("L")
        return img

    # ----------------------------------------------------------
    # Report loader
    # ----------------------------------------------------------
    def load_report(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ----------------------------------------------------------
    # __getitem__
    # ----------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        rpt_path = row["txt_file_path"]
        prompt   = row.get("view_prompt", "[VIEW=UNKNOWN]")

        if self.debug:
            print(f"[DEBUG] IMAGE : {img_path}")
            print(f"[DEBUG] REPORT: {rpt_path}")

        image  = self.load_png(img_path)
        report = self.load_report(rpt_path)

        return {
            "image"      : image,
            "view_prompt": prompt,
            "report"     : report,
            "image_id"   : row["image_id"],
            "txt_file"   : row["txt_file"],
            "xml_file"   : row.get("xml_file", ""),
        }

    def __len__(self) -> int:
        return len(self.df)


# ===============================================================
# LOAD METADATA, FILTER, SAMPLE TEST SUBSET
# ===============================================================
df = pd.read_csv(DATA_CSV)

# Filter for usable reports
df = df[(df["empty_findings"] == 0) & (df["report_exists"] == 1)].reset_index(drop=True)

# Add absolute report path
df["txt_file_path"] = df["txt_file"].apply(
    lambda x: os.path.join(REPORT_DIR, x) if isinstance(x, str) else None
)

# Add standardized view prompt
df["view_prompt"] = df["view"].apply(
    lambda v: f"[VIEW={v}]" if isinstance(v, str) else "[VIEW=UNKNOWN]"
)

# Sample 2,500 rows for testing
test_df = df.sample(n=2500, random_state=42).reset_index(drop=True)

# Optional train/val for completeness
remaining = df.drop(test_df.index).reset_index(drop=True)
train_df, val_df = train_test_split(remaining, test_size=0.1, random_state=42)

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

test_set = CXRReportDataset(test_df)
print("Datasets ready.")


# ===============================================================
# MODEL + TOKENIZER SETUP
# ===============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.padding_side = "left"     # decoder-only requirement

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float32,
).eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# ===============================================================
# LIGHTWEIGHT FREE-FORM POST-PROCESSING
# ===============================================================
def postprocess_freeform(text: str) -> str:
    """
    Removes SRRG-style tags, markdown, artifacts, and normalizes spacing.
    Conservative cleanup to preserve semantics.
    """
    if not text:
        return text

    text = re.sub(r"\[[^\]]+\]\s*", "", text)               # remove [Section: X]
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)            # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)                # italics
    text = text.replace("*", "")                            # stray
    text = re.sub(r"(^|\n)>\s*", r"\1", text)               # leading '>'
    text = text.replace("\n", " ")                          # newlines
    text = re.sub(r"\s*<\|box\|>.*?<\|/box\|>\s*", " ", text)
    text = re.sub(r"\s*<\|ref\|>\s*", "", text)
    text = re.sub(r"\s*<\|/ref\|>\s*", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


# ===============================================================
# SINGLE-IMAGE GENERATOR (tokenizer-only)
# ===============================================================
def generate_report_chexagent_tokenizer_only(
    pil_L_image: Image.Image,
    view_prompt: str,
    max_new_tokens: int = 160,
    do_sample: bool = False
) -> str:
    """
    Runs tokenizer-only CheXagent-2-3B generation for one image.
    Uses an on-disk JPEG; constructs multimodal chat template manually.
    """
    model.eval()

    # Anti-SRRG FINDINGS-only instruction
    instruction = (
        "You are a radiology assistant. Return ONLY the chest X-ray FINDINGS in 1–3 short sentences. "
        "Plain prose, no lists. DO NOT use any headings, labels, categories, or brackets. "
        "Do not prefix phrases with section names. Start directly with the findings."
    )

    # Save temporary image
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    pil_L_image.convert("RGB").save(tmp_path, "JPEG")

    # Build multimodal query for chat template
    query = tokenizer.from_list_format([
        {"image": tmp_path},
        {"text": f"{view_prompt} {instruction}"}
    ])

    conversation = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human",  "value": query},
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=(1 if do_sample else 3),
        temperature=(0.7 if do_sample else None),
        top_p=(0.9 if do_sample else None),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    # Remove prompt
    prompt_len = input_ids.shape[1]
    raw = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return postprocess_freeform(raw) or "[EMPTY]"


# ===============================================================
# BATCHED CHEXAGENT-2-3B INFERENCE (tokenizer-only)
# ===============================================================
def batched_chexagent2_3b_infer_tokenizer_only(
    dataset: Dataset,
    batch_size: int = 8,
    out_csv: str = OUT_CSV,
    max_new_tokens: int = 160,
    do_sample: bool = False,
    num_beams: int = 1
) -> pd.DataFrame:
    """
    Batched inference wrapper for CheXagent-2-3B in tokenizer-only mode.
    Creates temporary JPEG files, constructs multimodal chat prompts, pads
    sequences manually, runs generation, decodes continuation tokens, and
    writes results to CSV.
    """
    model.eval()
    rows, tmp_files = [], []
    N = len(dataset)

    print(f"Batched inference (CheXagent-2-3B tokenizer-only): {N} samples, batch_size={batch_size}")

    instruction = (
        "You are a radiology assistant. Return ONLY the chest X-ray FINDINGS in 1–3 short sentences. "
        "Plain prose, no lists. DO NOT use any headings, labels, categories, or brackets. "
        "Do not prefix phrases with section names. Start directly with the findings."
    )

    def chunker(it, size):
        for i in range(0, len(it), size):
            yield it[i:i+size]

    idxs = list(range(N))
    for chunk in tqdm(list(chunker(idxs, batch_size)), desc="Inferencing (batched)"):

        input_ids_list, metas = [], []

        # ------------------------------------------------------
        # Build per-sample chat prompts (each needs on-disk image)
        # ------------------------------------------------------
        for i in chunk:
            s = dataset[i]

            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            s["image"].convert("RGB").save(tmp_path, "JPEG")
            tmp_files.append(tmp_path)

            query = tokenizer.from_list_format([
                {"image": tmp_path},
                {"text": f"{s['view_prompt']} {instruction}"}
            ])

            conv = [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ]

            ids = tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            input_ids_list.append(ids)
            metas.append(s)

        # ------------------------------------------------------
        # Manual batch padding
        # ------------------------------------------------------
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_len = max(x.shape[1] for x in input_ids_list)
        batch_ids = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)

        for r, ids in enumerate(input_ids_list):
            batch_ids[r, :ids.shape[1]] = ids[0]

        batch_ids = batch_ids.to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=(1 if do_sample else num_beams),
            temperature=(0.7 if do_sample else None),
            top_p=(0.9 if do_sample else None),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        with torch.no_grad():
            out = model.generate(input_ids=batch_ids, **gen_kwargs)

        # ------------------------------------------------------
        # Decode each sample
        # ------------------------------------------------------
        for r, s in enumerate(metas):
            prompt_len = (batch_ids[r] != tokenizer.pad_token_id).sum().item()
            text = tokenizer.decode(out[r, prompt_len:], skip_special_tokens=True).strip()
            text = postprocess_freeform(text) or "[EMPTY]"

            rows.append({
                "image_id"   : str(s.get("image_id", "")),
                "xml_file"   : str(s.get("xml_file", "")),
                "view"       : str(s.get("view_prompt", "")),
                "report_gt"  : str(s.get("report", "")),
                "report_pred": text,
            })

        torch.cuda.empty_cache()

    # Cleanup all temp files
    for p in tmp_files:
        try:
            os.remove(p)
        except Exception:
            pass

    df_out = pd.DataFrame(rows, columns=["image_id","xml_file","view","report_gt","report_pred"])
    df_out.to_csv(out_csv, index=False)

    print(f"Saved CheXagent-2-3B predictions → {out_csv} (rows={len(df_out)})")
    return df_out


# ===============================================================
# MAIN ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    batched_chexagent2_3b_infer_tokenizer_only(
        test_set,
        batch_size=4,
        out_csv=OUT_CSV,
        max_new_tokens=160,
        do_sample=False,
        num_beams=3
    )
