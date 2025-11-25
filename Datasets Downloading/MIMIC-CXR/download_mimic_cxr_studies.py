"""
Description
-----------
This script downloads study folders (DICOMs + text reports) from the
MIMIC-CXR v2.1.0 PhysioNet repository using authenticated wget calls.
Studies are processed based on a CSV file containing study paths and a
downloaded flag. The script resumes downloads, skips completed studies,
and applies a cleanup step to flatten the directory structure after each
download.

Inputs
------
- studies-to-get-merged-with-path_v2.csv
    Must contain columns:
        dir_path : str  (relative PhysioNet folder path)
        downloaded : {0, 1}
    If "downloaded" is absent, it is initialized to 0.

Outputs
-------
- Downloaded study folders stored under ./files/
- Updated CSV with downloaded=1 for completed studies

Notes
-----
- The script preserves authentication details using user input.
- Flattening moves all nested files into the correct study directory.
- Directory structure is deterministic and consistent with PhysioNet.
"""

import os
import shutil
import getpass
import subprocess
import pandas as pd


# ======================================================================
# Load and prepare study list
# ======================================================================
csv_path = "studies-to-get-merged-with-path_v2.csv"
filtered_studies = pd.read_csv(csv_path)

# Ensure presence of downloaded flag
if "downloaded" not in filtered_studies.columns:
    filtered_studies["downloaded"] = 0

# Base directories and URL prefix
BASE_URL = "https://physionet.org/files/mimic-cxr/2.1.0/"
LOCAL_BASE = "files"
os.makedirs(LOCAL_BASE, exist_ok=True)


# ======================================================================
# Credential input
# ======================================================================
user = input("PhysioNet username: ")
password = getpass.getpass("PhysioNet password: ")


# ======================================================================
# Helper: Flatten nested downloaded folder structure
# ======================================================================
def flatten_study_folder(local_dir: str, patient_dir: str, study_id: str):
    """
    Moves all files from any nested subdirectories under `local_dir`
    directly into the final study directory. Also relocates the text
    report beside the study folder.

    Parameters
    ----------
    local_dir : str
        Temporary directory where wget places the downloaded structure.
    patient_dir : str
        Patient directory name (e.g., p10/p10000032).
    study_id : str
        Study folder name.
    """
    study_dir = os.path.join(LOCAL_BASE, patient_dir, study_id)
    os.makedirs(study_dir, exist_ok=True)

    # Collect all files from nested directories
    for root, _, files in os.walk(local_dir):
        for f in files:
            src = os.path.join(root, f)

            # Skip if file is already in target directory
            if os.path.samefile(os.path.dirname(src), study_dir):
                continue

            dst = os.path.join(study_dir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"   Move failed for {src}: {e}")

    # Remove empty directories left behind
    for root, dirs, _ in os.walk(local_dir, topdown=False):
        for d in dirs:
            path = os.path.join(root, d)
            if not os.listdir(path):
                os.rmdir(path)

    # Move text file beside the study folder (PhysioNet convention)
    txt_inside = os.path.join(study_dir, f"{study_id}.txt")
    txt_target = os.path.join(LOCAL_BASE, patient_dir, f"{study_id}.txt")

    if os.path.exists(txt_inside):
        shutil.move(txt_inside, txt_target)


# ======================================================================
# Sort studies: downloaded first (for resume behavior)
# ======================================================================
filtered_studies = filtered_studies.sort_values(
    by="downloaded",
    ascending=False
).reset_index(drop=True)

unique_dirs = filtered_studies["dir_path"].dropna().tolist()
print(f"Unique study folders: {len(unique_dirs)}")


# ======================================================================
# Main download loop
# ======================================================================
for i, d in enumerate(unique_dirs, start=1):
    mask = filtered_studies["dir_path"] == d
    is_downloaded = filtered_studies.loc[mask, "downloaded"].iloc[0]

    # Skip if already downloaded
    if is_downloaded == 1:
        print(f"[{i}/{len(unique_dirs)}] Skipping (already downloaded): {d}")
        continue

    print(f"[{i}/{len(unique_dirs)}] Downloading: {d}")

    # Parse folder structure components
    rel_path = d.strip("/")
    study_id = os.path.basename(rel_path)
    patient_dir = os.path.dirname(rel_path)

    folder_url = BASE_URL + rel_path + "/"
    txt_url = BASE_URL + rel_path + ".txt"

    local_dir = os.path.join(LOCAL_BASE, rel_path)
    os.makedirs(local_dir, exist_ok=True)

    try:
        # --------------------------------------------------------------
        # Step 1: Download DICOM folder recursively
        # --------------------------------------------------------------
        cmd_folder = [
            "wget", "-r", "-N", "-c", "-np",
            "--user", user, "--password", password,
            "-P", local_dir, folder_url
        ]
        subprocess.run(
            cmd_folder,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # --------------------------------------------------------------
        # Step 2: Download study-level text report
        # --------------------------------------------------------------
        cmd_txt = f"""
        wget -q --no-verbose --progress=dot:giga \
        --no-check-certificate --auth-no-challenge \
        --user "{user}" --password "{password}" \
        -P "{local_dir}" "{txt_url}" >/dev/null 2>&1
        """
        subprocess.run(
            cmd_txt,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # --------------------------------------------------------------
        # Step 3: Flatten the downloaded directory structure
        # --------------------------------------------------------------
        print(f"   Flattening folder for {study_id} ...")
        flatten_study_folder(local_dir, patient_dir, study_id)

        # --------------------------------------------------------------
        # Step 4: Mark completed
        # --------------------------------------------------------------
        filtered_studies.loc[mask, "downloaded"] = 1
        filtered_studies.to_csv(csv_path, index=False)
        print(f"Completed: {rel_path}\n")

    except subprocess.CalledProcessError as e:
        print(f"Error during download: {rel_path} -> {e}")
        filtered_studies.to_csv(csv_path, index=False)
        continue


# ======================================================================
# Final summary
# ======================================================================
print("All downloads completed.")
print(f"{filtered_studies['downloaded'].sum()} / {len(filtered_studies)} studies downloaded successfully.")
