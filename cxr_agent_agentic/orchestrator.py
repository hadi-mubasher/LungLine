"""High-level orchestration of tools for the Gradio UI.

The main entry point here is :func:`chat_handler`, which receives
the Gradio chat state + image + registry state, routes the user
message to the appropriate tool using GPT-4o-mini (or a keyword
fallback), and returns updated outputs for the UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from config import LABEL_COLS
from modules.qc import load_cxr_from_path, run_quality_check, format_qc_message
from modules.classifier import load_classifier_model, run_classifier, summarize_probs
from modules.report import generate_cxr_report
from modules.heatmaps import generate_heatmaps_for_top3
from modules.db import (
    save_study_to_db,
    retrieve_patient_history,
    fetch_patient_reports_for_patient_id,
)
from modules.agent_tools import (
    choose_tool_via_gpt,
    explanation_mode_answer,
    extract_patient_id_from_text,
    summarize_reports_for_patient_llm,
)
from modules.guardrails import moderate_content  # if not present, we add this


# Load heavy models once at module import
cxr_classifier = load_classifier_model()


def ensure_probs_dict(probs_dict: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Utility: ensure we always have a dict for probabilities."""
    return dict(probs_dict or {})


def chat_handler(
    message: str,
    history: List[Dict[str, str]],
    explanation_mode: str,
    image_path: Optional[str],
    qc_dict: Optional[Dict[str, Any]],
    probs_dict: Optional[Dict[str, float]],
    report_text: str,
    current_heatmap_gallery,
    current_heatmap_status: str,
    current_patient_id: Optional[str],
):
    """Core chat handler called from the Gradio UI."""
    history = history or []
    qc_dict = qc_dict or {}
    probs_dict = ensure_probs_dict(probs_dict)
    report_text = report_text or ""
    current_heatmap_gallery = current_heatmap_gallery or []

    # Append user message to chat history
    if message:
        history.append({"role": "user", "content": message})
    else:
        probs_rows = [
            [label, float(p)]
            for label, p in sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            history,
            explanation_mode,
            qc_dict,
            probs_dict,
            report_text,
            qc_dict,
            probs_rows,
            report_text,
            current_heatmap_gallery,
            current_heatmap_status,
        )

    if image_path is None:
        assistant_reply = (
            "Please upload a chest X-ray (PNG/JPEG or DICOM) before asking for a diagnosis, "
            "report, heatmap, or student-mode explanation."
        )
        history.append({"role": "assistant", "content": assistant_reply})
        probs_rows = [
            [label, float(p)]
            for label, p in sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            history,
            explanation_mode,
            qc_dict,
            probs_dict,
            report_text,
            qc_dict,
            probs_rows,
            report_text,
            current_heatmap_gallery,
            current_heatmap_status,
        )

    # Run QC for the current image (each chat step)
    image, dicom_ds = load_cxr_from_path(image_path)
    qc_res = run_quality_check(image, dicom_ds)
    qc_dict_new = qc_res.to_dict()
    qc_msg_full = format_qc_message(qc_res)
    qc_short = (
        f"QC status for this image: **{qc_res.severity.upper()}** "
        "(see QC panel on the right for details)."
    )
    # GUARDRAILS (Moderation + Rule-based)
    guard = moderate_content(message)

    if not guard["safe"]:
        assistant_reply = (
            f"⚠️ I cannot process this request.\n"
            f"Reason: {guard['reason']}\n\n"
            "Please keep questions safe and focused on chest X-ray interpretation."
        )
        history.append({"role": "assistant", "content": assistant_reply})

        # Return unchanged state
        probs_rows = [
            [label, float(p)]
            for label, p in sorted((probs_dict or {}).items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            history,
            explanation_mode,
            qc_dict_new,
            probs_dict,
            report_text,
            qc_dict_new,
            probs_rows,
            report_text,
            current_heatmap_gallery,
            current_heatmap_status,
        )


    # Use GPT-4o-mini (or fallback) to choose which tool to run
    tool_name = choose_tool_via_gpt(message, image_available=(image_path is not None))

    # If QC fails, gate interpretation tools
    if qc_res.severity == "fail" and tool_name in {"diagnosis", "report", "heatmap", "student"}:
        assistant_reply = (
            qc_msg_full
            + "\n\nBecause QC failed, I am not running automated diagnosis, report generation, "
            "heatmaps, or tutoring on this image. Please upload a better-quality chest X-ray."
        )
        history.append({"role": "assistant", "content": assistant_reply})
        probs_rows = [
            [label, float(p)]
            for label, p in sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            history,
            explanation_mode,
            qc_dict_new,
            probs_dict,
            report_text,
            qc_dict_new,
            probs_rows,
            report_text,
            current_heatmap_gallery,
            current_heatmap_status,
        )

    new_probs = dict(probs_dict)
    new_report = report_text
    heatmap_gallery_out = current_heatmap_gallery
    heatmap_status_out = current_heatmap_status

    # ------------------------------------------------------------------
    # Tool: patient_reports
    # ------------------------------------------------------------------
    if tool_name == "patient_reports":
        pid_text = extract_patient_id_from_text(message)
        if not pid_text and current_patient_id:
            pid_text = str(current_patient_id).strip()

        if not pid_text:
            assistant_reply = (
                "To summarise prior reports, please either:\n"
                "- enter a valid `patient_id` in the **Patient Registry** and click *Retrieve*, or\n"
                "- mention the patient id explicitly in your question, e.g. "
                '"give me summaries of the reports for patient 12".'
            )
            history.append({"role": "assistant", "content": assistant_reply})
            probs_rows = [
                [label, float(p)]
                for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
            ]
            return (
                history,
                explanation_mode,
                qc_dict_new,
                new_probs,
                new_report,
                qc_dict_new,
                probs_rows,
                new_report,
                heatmap_gallery_out,
                heatmap_status_out,
            )

        patient, report_rows = fetch_patient_reports_for_patient_id(pid_text)
        if patient is None or not report_rows:
            assistant_reply = (
                f"I couldn't find any saved reports in the database for patient_id `{pid_text}`. "
                "Make sure you have previously saved this patient via the Patient Registry module."
            )
            history.append({"role": "assistant", "content": assistant_reply})
            probs_rows = [
                [label, float(p)]
                for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
            ]
            return (
                history,
                explanation_mode,
                qc_dict_new,
                new_probs,
                new_report,
                qc_dict_new,
                probs_rows,
                new_report,
                heatmap_gallery_out,
                heatmap_status_out,
            )

        summaries_md = summarize_reports_for_patient_llm(str(patient.id), report_rows)
        assistant_reply = (
            f"Here are summaries of the reports I have stored for patient_id `{patient.id}`:\n\n"
            + summaries_md
        )
        history.append({"role": "assistant", "content": assistant_reply})
        probs_rows = [
            [label, float(p)]
            for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            history,
            explanation_mode,
            qc_dict_new,
            new_probs,
            new_report,
            qc_dict_new,
            probs_rows,
            new_report,
            heatmap_gallery_out,
            heatmap_status_out,
        )

    # ------------------------------------------------------------------
    # Tool: diagnosis
    # ------------------------------------------------------------------
    if tool_name == "diagnosis":
        if not new_probs:
            new_probs = run_classifier(cxr_classifier, image)

        probs_rows = [
            [label, float(p)]
            for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
        ]

        diag_text = "### AI diagnosis (classification probabilities)\n"
        diag_text += summarize_probs(new_probs, top_k=7) if new_probs else "No outputs."

        assistant_reply = qc_msg_full + "\n\n" + diag_text
        history.append({"role": "assistant", "content": assistant_reply})

        return (
            history,
            explanation_mode,
            qc_dict_new,
            new_probs,
            new_report,
            qc_dict_new,
            probs_rows,
            new_report,
            heatmap_gallery_out,
            heatmap_status_out,
        )

    # ------------------------------------------------------------------
    # Tool: report
    # ------------------------------------------------------------------
    if tool_name == "report":
        if not new_report:
            new_report = generate_cxr_report(image)

        probs_rows = [
            [label, float(p)]
            for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
        ]

        report_text_block = "### AI-generated CXR report\n" + new_report.strip()
        assistant_reply = qc_msg_full + "\n\n" + report_text_block
        history.append({"role": "assistant", "content": assistant_reply})

        return (
            history,
            explanation_mode,
            qc_dict_new,
            new_probs,
            new_report,
            qc_dict_new,
            probs_rows,
            new_report,
            heatmap_gallery_out,
            heatmap_status_out,
        )

    # ------------------------------------------------------------------
    # Tool: heatmap
    # ------------------------------------------------------------------
    if tool_name == "heatmap":
        gallery_items, heatmap_status_out, new_probs = generate_heatmaps_for_top3(
            image_path,
            image_loader=load_cxr_from_path,
            probs=new_probs,
            classifier_fn=lambda img: run_classifier(cxr_classifier, img),
        )
        heatmap_gallery_out = gallery_items

        probs_rows = [
            [label, float(p)]
            for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
        ]

        assistant_reply = qc_msg_full + "\n\n" + heatmap_status_out
        history.append({"role": "assistant", "content": assistant_reply})

        return (
            history,
            explanation_mode,
            qc_dict_new,
            new_probs,
            new_report,
            qc_dict_new,
            probs_rows,
            new_report,
            heatmap_gallery_out,
            heatmap_status_out,
        )

    # ------------------------------------------------------------------
    # Tool: offtopic
    # ------------------------------------------------------------------
    if tool_name == "offtopic":
        assistant_reply = (
            "I’m focused on this chest X-ray only. Please ask questions related to the current image, "
            "such as diagnosis, report, heatmaps, or imaging-based teaching."
        )
        history.append({"role": "assistant", "content": assistant_reply})
        probs_rows = [
            [label, float(p)]
            for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            history,
            explanation_mode,
            qc_dict_new,
            new_probs,
            new_report,
            qc_dict_new,
            probs_rows,
            new_report,
            heatmap_gallery_out,
            heatmap_status_out,
        )

    # ------------------------------------------------------------------
    # Tool: student (teaching mode)
    # ------------------------------------------------------------------
    if not new_report:
        assistant_reply = (
            qc_short
            + "\n\nTo discuss this case in more detail, I need an AI-generated report first. "
            "Please ask me to 'generate a report' for this X-ray, then you can ask follow-up questions."
        )
        history.append({"role": "assistant", "content": assistant_reply})
        probs_rows = [
            [label, float(p)]
            for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return (
            history,
            explanation_mode,
            qc_dict_new,
            new_probs,
            new_report,
            qc_dict_new,
            probs_rows,
            new_report,
            heatmap_gallery_out,
            heatmap_status_out,
        )

    if explanation_mode == "Student":
        student_answer = explanation_mode_answer(
            question=message,
            mode="Student",
            report_text=new_report,
            probs=new_probs,
        )
        assistant_reply = qc_short + "\n\n" + student_answer
    else:
        assistant_reply = qc_short + "\n\n" + explanation_mode_answer(
            question=message,
            mode="Clinician",
            report_text=new_report,
            probs=new_probs,
        )

    history.append({"role": "assistant", "content": assistant_reply})

    probs_rows = [
        [label, float(p)]
        for label, p in sorted(new_probs.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return (
        history,
        explanation_mode,
        qc_dict_new,
        new_probs,
        new_report,
        qc_dict_new,
        probs_rows,
        new_report,
        heatmap_gallery_out,
        heatmap_status_out,
    )


# ----------------------------------------------------------------------
# Patient registry handlers used by the UI
# ----------------------------------------------------------------------
def insert_to_registry(
    patient_id_text,
    name,
    age,
    gender,
    image_path,
    qc_dict,
    probs_dict,
    report_text,
):
    """Insert current study into the DB and return UI updates."""
    try:
        pid, sid, history_rows = save_study_to_db(
            patient_id_text,
            name,
            age,
            gender,
            image_path,
            qc_dict,
            probs_dict,
            report_text,
        )
        status = (
            f"✅ Saved study to DB.\n\n"
            f"- **patient_id:** `{pid}`\n"
            f"- **study_id:** `{sid}`\n"
            f"- Entries in history: {len(history_rows)}"
        )
        table_rows = [
            [
                row.get("study_id", ""),
                row.get("created_at", ""),
                row.get("image_hash", ""),
                row.get("top_probs", ""),
                row.get("impression", ""),
            ]
            for row in history_rows
        ]
        return status, pid, sid, table_rows, history_rows, None
    except Exception as e:  # pragma: no cover - defensive
        return f"⚠️ Failed to save study: {e}", "", "", [], [], None


def retrieve_from_registry(
    patient_id_text,
    image_path,
):
    """Retrieve an existing patient + history from the DB."""
    try:
        patient, created_new, matched_on, history_rows = retrieve_patient_history(
            patient_id_text, image_path
        )

        if patient is None:
            status = (
                "❌ No existing records found in the database.\n\n"
                "Please enter a **correct patient_id** for a registered patient "
                "or ensure this CXR belongs to a patient already saved in the system."
            )
            return status, "", "", [], [], None

        status = (
            f"✅ Found patient in DB (matched on **{matched_on}**).\n\n"
            f"- **patient_id:** `{patient.id}`\n"
            f"- Studies in history: {len(history_rows)}"
        )
        table_rows = [
            [
                row.get("study_id", ""),
                row.get("created_at", ""),
                row.get("image_hash", ""),
                row.get("top_probs", ""),
                row.get("impression", ""),
            ]
            for row in history_rows
        ]
        return status, str(patient.id), "", table_rows, history_rows, None

    except Exception as e:  # pragma: no cover - defensive
        return f"⚠️ Failed to retrieve history: {e}", "", "", [], [], None


def on_history_select(table_data, history_rows, evt):
    """Return an image for the selected history row in the UI."""
    import os
    from modules.qc import load_cxr_from_path

    if history_rows is None:
        return None

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        row = history_rows[row_idx]
    except (IndexError, TypeError):
        return None

    img_path = row.get("image_path", "")
    if not img_path or not os.path.exists(img_path):
        return None

    img, _ = load_cxr_from_path(img_path)
    return img
