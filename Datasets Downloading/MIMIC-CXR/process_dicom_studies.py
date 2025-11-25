"""
Description
-----------
This script processes DICOM studies that were downloaded from PhysioNet,
converts each DICOM to PNG, extracts non-identifiable metadata, and stores
all output in structured directories. The script supports parallel
processing across CPU cores and incremental resumption via a tracking CSV.

Inputs
------
- studies-to-get-merged-with-path_v2.csv
  Contains: subject_id, study_id, dir_path, downloaded, processed

Outputs
-------
- PNG images stored in IMAGES_BASE/<patient_folder>/
- Metadata rows appended to dcm_info.csv
- Updated studies-to-get-merged-with-path_processing_v2.csv

Notes
-----
- All DICOM failures are safely skipped.
- Normalization of pixel data uses min-max scaling and 8-bit output.
- Parallel processing uses ProcessPoolExecutor for speed.
"""

import os
import glob
import pandas as pd
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ===============================================================
# CONFIGURATION
# ===============================================================
SRC_CSV = "studies-to-get-merged-with-path_v2.csv"
PROC_CSV = "studies-to-get-merged-with-path_processing_v2.csv"
DCM_INFO_CSV = "dcm_info.csv"
IMAGES_BASE = "images"

# Use all CPU cores except one (fallback to 1 if cpu_count() returns None)
MAX_WORKERS = os.cpu_count() - 1 or 1


# ===============================================================
# INITIALIZATION
# ===============================================================
# Create or load the processing-tracking CSV
if not os.path.exists(PROC_CSV):
    df = pd.read_csv(SRC_CSV)
    if "processed" not in df.columns:
        df["processed"] = 0
    df.to_csv(PROC_CSV, index=False)
else:
    df = pd.read_csv(PROC_CSV)
    if "processed" not in df.columns:
        df["processed"] = 0

# Ensure image directory exists
os.makedirs(IMAGES_BASE, exist_ok=True)


# ===============================================================
# UTILITY: Save DICOM image as PNG
# ===============================================================
def save_dicom_as_png(ds, out_path):
    """
    Convert a DICOM dataset's pixel_array to an 8-bit PNG file.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset
        Loaded DICOM dataset.
    out_path : str
        Output location for the PNG file.

    Returns
    -------
    bool
        True if written successfully, False otherwise.
    """
    try:
        arr = ds.pixel_array.astype(float)

        # Normalize to [0, 255]
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255.0
        arr = arr.astype(np.uint8)

        Image.fromarray(arr).save(out_path)
        return True
    except Exception:
        return False


# ===============================================================
# UTILITY: Extract metadata
# ===============================================================
def extract_dicom_info(ds, subject_id, study_id, dcm_path, image_path):
    """
    Extract non-identifiable metadata fields from a DICOM dataset.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset
        Loaded DICOM dataset.
    subject_id : str/int
    study_id : str/int
    dcm_path : str
    image_path : str

    Returns
    -------
    dict
        Dictionary of selected metadata fields.
    """
    def get(tag, default=None):
        return getattr(ds, tag, default)

    return {
        "subject_id": subject_id,
        "study_id": study_id,
        "dcm_path": dcm_path,
        "image_path": image_path,
        "SOPInstanceUID": get("SOPInstanceUID"),
        "ViewPosition": get("ViewPosition"),
        "BodyPartExamined": get("BodyPartExamined"),
        "PixelSpacing": get("PixelSpacing"),
        "Rows": get("Rows"),
        "Columns": get("Columns"),
        "ExposureIndex": get("ExposureIndex"),
        "DeviationIndex": get("DeviationIndex"),
        "StudyDate": get("StudyDate"),
        "Manufacturer": get("Manufacturer"),
        "PhotometricInterpretation": get("PhotometricInterpretation"),
        "ImageLaterality": get("ImageLaterality"),
    }


# ===============================================================
# WORKER: Process all DICOM files for one study
# ===============================================================
def process_study(row):
    """
    Worker function executed in parallel.

    Parameters
    ----------
    row : dict
        One study row containing at least: subject_id, study_id, dir_path.

    Returns
    -------
    tuple
        (study_id, list_of_metadata_dicts)
    """
    subject_id = row["subject_id"]
    study_id = row["study_id"]
    dir_path = os.path.join("files", row["dir_path"])

    # Skip if directory does not exist
    if not os.path.isdir(dir_path):
        return study_id, []

    # Find all DICOM files within the study directory
    dcm_files = glob.glob(os.path.join(dir_path, "*.dcm"))
    if not dcm_files:
        return study_id, []

    study_results = []

    for dcm_path in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_path)
        except (InvalidDicomError, FileNotFoundError):
            continue

        # Determine output folder (patient-level)
        # Example path should include patient folder e.g. images/p10/
        subfolder = os.path.basename(os.path.dirname(os.path.dirname(dcm_path)))
        img_dir = os.path.join(IMAGES_BASE, subfolder)
        os.makedirs(img_dir, exist_ok=True)

        # Output filename: <subject>_<study>_<view>.png
        view = getattr(ds, "ViewPosition", "UNK")
        img_name = f"{subject_id}_{study_id}_{view}.png"
        img_path = os.path.join(img_dir, img_name)

        if save_dicom_as_png(ds, img_path):
            info = extract_dicom_info(ds, subject_id, study_id, dcm_path, img_path)
            study_results.append(info)

    return study_id, study_results


# ===============================================================
# PREPARE METADATA OUTPUT CSV
# ===============================================================
if not os.path.exists(DCM_INFO_CSV):
    pd.DataFrame(columns=[
        "subject_id", "study_id", "dcm_path", "image_path", "SOPInstanceUID",
        "ViewPosition", "BodyPartExamined", "PixelSpacing", "Rows", "Columns",
        "ExposureIndex", "DeviationIndex", "StudyDate", "Manufacturer",
        "PhotometricInterpretation", "ImageLaterality"
    ]).to_csv(DCM_INFO_CSV, index=False)


# ===============================================================
# MAIN PARALLEL EXECUTION
# ===============================================================
to_process = df[df["processed"] == 0].to_dict(orient="records")

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_study, row): row for row in to_process}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM studies"):
        study_id, results = future.result()

        # Append metadata if present
        if results:
            pd.DataFrame(results).to_csv(DCM_INFO_CSV, mode="a", index=False, header=False)

        # Mark study as processed
        df.loc[df["study_id"] == study_id, "processed"] = 1
        df.to_csv(PROC_CSV, index=False)


print("Parallel DICOM processing complete.")
