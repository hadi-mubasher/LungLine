"""
Description
-----------
This script processes raw radiology reports for the CXR project and
extracts only the radiographic findings using an OpenAI model
(gpt-4o-mini). The same pipeline is applied to three dataset splits:
train, validation, and test.

For each split:
- Reads an input CSV containing subject_id, study_id, and report_path.
- Maintains a corresponding "processed" CSV with a processed flag
  to allow resuming after interruptions.
- Normalizes report paths by stripping a prefix (as used in the
  original notebooks).
- Loads each raw report from disk.
- Calls the OpenAI API with a radiology text filter system prompt to
  return only CXR-observable findings.
- Writes the cleaned findings to a per-study text file under
  Dataset/reports.
- Updates the processed flag in the CSV after each successful row.

Inputs
------
- Dataset/cxr_reports_train.csv
- Dataset/cxr_reports_val.csv
- Dataset/cxr_reports_test.csv

Outputs
-------
- Dataset/cxr_reports_train_processed.csv
- Dataset/cxr_reports_val_processed.csv
- Dataset/cxr_reports_test_processed.csv
- Dataset/reports/{subject_id}_{study_id}.txt

Notes
-----
- API key is loaded from google.colab.userdata (Colab environment).
- The script is resume-friendly via the "processed" column.
- Radiology content extraction is handled by the OpenAI chat API.
"""

import os
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import openai
from openai import OpenAI
from google.colab import userdata  # Colab-based secret storage


# ===============================================================
# CONFIGURATION
# ===============================================================
BASE = Path("Dataset")
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Split-specific configuration (matches original notebooks)
SPLITS = [
    {
        "name": "train",
        "input_csv": BASE / "cxr_reports_train.csv",
        "processed_csv": BASE / "cxr_reports_train_processed.csv",
        "path_prefix_to_strip": "/content/drive/MyDrive/Dataset/",
    },
    {
        "name": "val",
        "input_csv": BASE / "cxr_reports_val.csv",
        "processed_csv": BASE / "cxr_reports_val_processed.csv",
        "path_prefix_to_strip": "dataset downloading/Dataset/",
    },
    {
        "name": "test",
        "input_csv": BASE / "cxr_reports_test.csv",
        "processed_csv": BASE / "cxr_reports_test_processed.csv",
        "path_prefix_to_strip": "dataset downloading/Dataset/",
    },
]


# ===============================================================
# OPENAI CLIENT AND PROMPT
# ===============================================================
# Client initialization (API key loaded from Colab userdata)
client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a radiology text filter.

The text provided above is a full radiology report.

Rewrite it to retain only descriptive information that can be directly observed on a chest X-ray (CXR), such as findings related to the lungs, heart, bones, pleura, medical devices, or visible anatomy.

Do not include:
- Patient history, symptoms, or reasons for examination
- Clinical impressions, interpretations, or conclusions
- Notifications, timestamps, or physician references
- Placeholders or missing information (for example, “_”)

Include only the radiographic findings as a single coherent paragraph.
Preserve all medical terminology and descriptive details exactly as they appear (for example, “atelectasis”, “pleural effusion”, “cardiomegaly”).

If the report does not contain any radiographic findings, respond with:
None
"""


# ===============================================================
# HELPERS
# ===============================================================
def read_report_text(path_str: str) -> str:
    """
    Load raw report text from disk.

    Parameters
    ----------
    path_str : str
        Relative or absolute path to the report text file.

    Returns
    -------
    str
        Report text stripped of leading/trailing whitespace.

    Raises
    ------
    FileNotFoundError
        If the path does not exist on disk.
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = BASE / path_str
    if not p.exists():
        raise FileNotFoundError(f"Report not found: {p}")
    try:
        return p.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return p.read_text(errors="ignore").strip()


def extract_cxr_findings(text: str, max_retries: int = 4, backoff: float = 3.0) -> str:
    """
    Call gpt-4o-mini to extract only CXR-observable findings.

    Parameters
    ----------
    text : str
        Full radiology report.
    max_retries : int, optional
        Maximum number of retries on transient errors.
    backoff : float, optional
        Base backoff in seconds for exponential retry.

    Returns
    -------
    str
        Cleaned findings text or "None" if no findings.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            out = resp.choices[0].message.content.strip()
            if not out:
                return "None"
            if out.lower() in {"none", "no findings", "no radiographic findings"}:
                return "None"
            return out
        except Exception as e:
            last_err = e
            sleep_s = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


def prepare_split_dataframe(input_csv: Path, processed_csv: Path, path_prefix_to_strip: str) -> pd.DataFrame:
    """
    Initialize or load the processed CSV for a split and normalize paths.

    Parameters
    ----------
    input_csv : Path
        Path to the original split CSV (train/val/test).
    processed_csv : Path
        Path to the resume-friendly CSV with 'processed' flag.
    path_prefix_to_strip : str
        Prefix to remove from report_path, as used in original notebooks.

    Returns
    -------
    pd.DataFrame
        Dataframe containing subject_id, study_id, report_path, processed.
    """
    if processed_csv.exists():
        df = pd.read_csv(processed_csv)
        needed = ["subject_id", "study_id", "report_path", "processed"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in existing {processed_csv.name}: {missing}")
    else:
        base_df = pd.read_csv(input_csv)
        keep_cols = ["subject_id", "study_id", "report_path"]
        for c in keep_cols:
            if c not in base_df.columns:
                raise ValueError(f"Column '{c}' not found in {input_csv.name}")
        df = base_df[keep_cols].copy()
        df["processed"] = 0
        df.to_csv(processed_csv, index=False)

    # Normalize report paths by stripping the given prefix (same as notebooks)
    df["report_path"] = df["report_path"].astype(str).str.replace(path_prefix_to_strip, "")

    return df


def process_split(split_cfg: dict) -> None:
    """
    Run the CXR findings extraction pipeline for a single split.

    Parameters
    ----------
    split_cfg : dict
        Configuration dictionary with keys:
        - name
        - input_csv
        - processed_csv
        - path_prefix_to_strip
    """
    name = split_cfg["name"]
    input_csv = split_cfg["input_csv"]
    processed_csv = split_cfg["processed_csv"]
    path_prefix = split_cfg["path_prefix_to_strip"]

    print("\n===================================================")
    print(f"Processing split: {name}")
    print("Input CSV: ", input_csv)
    print("Output CSV:", processed_csv)
    print("Reports dir:", REPORTS_DIR)

    df = prepare_split_dataframe(input_csv, processed_csv, path_prefix)

    # Select only unprocessed rows
    todo_mask = df["processed"] == 0
    to_process = df[todo_mask].copy()
    print(f"Unprocessed rows: {len(to_process)}")

    # Iterate rows and process one by one with checkpointing
    for idx, row in tqdm(
        to_process.iterrows(),
        total=len(to_process),
        desc=f"Processing ({name})"
    ):
        subj = str(row["subject_id"])
        study = str(row["study_id"])
        rpath = str(row["report_path"])

        out_path = REPORTS_DIR / f"{subj}_{study}.txt"

        # If output already exists, mark as processed and continue (idempotent)
        if out_path.exists():
            df.loc[idx, "processed"] = 1
            df.to_csv(processed_csv, index=False)
            continue

        try:
            raw = read_report_text(rpath)
            if not raw:
                out_path.write_text("None", encoding="utf-8")
                df.loc[idx, "processed"] = 1
                df.to_csv(processed_csv, index=False)
                continue

            cleaned = extract_cxr_findings(raw)

            # Save cleaned findings
            out_path.write_text(cleaned, encoding="utf-8")

            # Mark as processed and checkpoint
            df.loc[idx, "processed"] = 1
            df.to_csv(processed_csv, index=False)

        except FileNotFoundError as e:
            # Leave processed=0 so paths can be fixed later
            print(f"[Missing] {subj}_{study} -> {e}")
        except Exception as e:
            # API or other failure; keep processed=0 for resume
            print(f"[Error] {subj}_{study}: {e}")


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    for split_cfg in SPLITS:
        process_split(split_cfg)
