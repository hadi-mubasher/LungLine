"""Agentic tool schema and GPT-4o-mini routing helpers.

This module defines:
  - A simple tool specification dataclass.
  - Helper functions to detect simple intents as a fallback.
  - GPT-4o-mini based routing that chooses which tool to apply
    to a given user message.
  - Higher-level explanation and patient-report-summarisation tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
from openai import OpenAI

from config import OPENAI_MODEL_NAME, OPENAI_API_KEY_ENV
from .classifier import summarize_probs


# -------------------------------------------------------------------------
# OpenAI client initialisation
# -------------------------------------------------------------------------
def get_openai_client() -> Optional[OpenAI]:
    """Safely construct an OpenAI client if the API key is present."""
    import os

    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        print("⚠️ OPENAI_API_KEY not set – GPT-4o-mini features disabled.")
        return None
    try:
        client = OpenAI()
        print("OpenAI client initialised.")
        return client
    except Exception as exc:  # pragma: no cover - defensive
        print("Failed to create OpenAI client:", exc)
        return None


openai_client: Optional[OpenAI] = get_openai_client()


@dataclass
class ToolSpec:
    """Lightweight description of an available tool for the agent."""

    name: str
    description: str
    # For simplicity, arguments are documented in description; parsing
    # and mapping are handled in the orchestrator.
    requires_image: bool = True


# These tool names mirror the original manual intents
TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="diagnosis",
        description=(
            "Run the CNN classifier and return diagnosis probabilities "
            "for common chest X-ray labels."
        ),
    ),
    ToolSpec(
        name="report",
        description=(
            "Generate a full chest X-ray report using the vision-language model."
        ),
    ),
    ToolSpec(
        name="heatmap",
        description=(
            "Generate Grad-CAM-like heatmaps for the top 3 localisation labels."
        ),
    ),
    ToolSpec(
        name="student",
        description=(
            "Answer in a teaching / student-friendly way, explaining the "
            "findings and differential using the existing report + probabilities."
        ),
    ),
    ToolSpec(
        name="patient_reports",
        description=(
            "Summarise all previous reports for this patient from the database."
        ),
        requires_image=False,
    ),
    ToolSpec(
        name="offtopic",
        description=(
            "Used when the question is off-topic or unrelated to the current CXR."
        ),
        requires_image=False,
    ),
]


# -------------------------------------------------------------------------
# Simple keyword-based intent detection (fallback when GPT is off)
# -------------------------------------------------------------------------
ON_TOPIC_KEYWORDS = [
    # Anatomy / modality
    "lung", "lungs", "chest", "thorax",
    "cxr", "xray", "x-ray", "x ray",
    # Findings / labels
    "pleural", "effusion", "pneumonia", "atelectasis",
    "pneumothorax", "lesion", "opacity", "fracture",
    "mediastinum", "cardiomegaly", "support device",
    "enlarged cardiomediastinum",
    # AI outputs
    "report", "impression", "finding", "findings",
    "diagnosis", "diagnose", "diagnostic", "diagnos",
    "classification", "classify", "labels", "probabilities",
]


def is_on_topic(question: str) -> bool:
    """Return True if the question is within the CXR / thoracic domain."""
    q = question.lower()
    return any(kw in q for kw in ON_TOPIC_KEYWORDS)


def detect_intent_fallback(message: str) -> str:
    """Very simple keyword-based intent classifier (manual fallback)."""
    q = message.lower().strip()

    explain_keywords = [
        "explain", "explanation",
        "clarify", "clarification",
        "simplify", "simple terms",
        "layman", "teach", "learning",
        "difference between", "how is", "how are",
        "why is", "why does",
    ]

    if any(ek in q for ek in explain_keywords):
        if any(
            kw in q
            for kw in ["report", "diagnos", "finding", "cxr", "xray", "x-ray", "x ray"]
        ):
            return "student"

    diag_keywords = [
        "diagnosis", "diagnose", "dx",
        "classification", "classify",
        "labels", "label this", "probabilities",
    ]
    if any(k in q for k in diag_keywords):
        return "diagnosis"

    if "diagnos" in q and any(kw in q for kw in ["cxr", "xray", "x-ray", "x ray", "chest"]):
        return "diagnosis"

    if "report" in q:
        action_keywords = [
            "generate", "write", "create", "make",
            "produce", "give", "show", "provide",
        ]
        if any(ak in q for ak in action_keywords):
            return "report"

    heatmap_keywords = [
        "heatmap", "heat map",
        "grad-cam", "gradcam",
        "saliency", "attention map",
    ]
    if any(k in q for k in heatmap_keywords):
        return "heatmap"

    patient_keywords = ["patient", "patient_id", "patient id"]
    summary_keywords = [
        "summary", "summaries", "summarise", "summarize",
        "previous reports", "past reports", "all reports", "history",
    ]
    if any(pk in q for pk in patient_keywords) and any(sk in q for sk in summary_keywords):
        return "patient_reports"

    if is_on_topic(message):
        return "student"

    return "offtopic"


# -------------------------------------------------------------------------
# GPT-4o-mini tool selection
# -------------------------------------------------------------------------
def choose_tool_via_gpt(message: str, image_available: bool) -> str:
    """Ask GPT-4o-mini to choose one of the TOOL_SPECS tools."""
    if openai_client is None:
        # Fallback to keyword-based classifier
        return detect_intent_fallback(message)

    tools_desc = [
        {
            "name": t.name,
            "description": t.description,
            "requires_image": t.requires_image,
        }
        for t in TOOL_SPECS
    ]

    system_prompt = (
        "You are a router for a chest X-ray assistant. "
        "Given a user message and a list of tools, choose exactly ONE tool name "
        "from the list that best answers the user. "
        "Return a JSON object of the form {\"tool\": \"name\"} with no extra text."
    )

    user_content = {
        "message": message,
        "image_available": image_available,
        "tools": tools_desc,
    }

    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=50,
    )

    try:
        data = json.loads(completion.choices[0].message.content)
        tool_name = data.get("tool", "")
        if tool_name:
            return tool_name
    except Exception:
        pass

    # If anything goes wrong, fall back to keyword rules
    return detect_intent_fallback(message)


# -------------------------------------------------------------------------
# Helper: extract patient_id from free text
# -------------------------------------------------------------------------
import re


def extract_patient_id_from_text(message: str) -> Optional[str]:
    """Try to extract a patient_id mentioned in free-text."""
    q = message.lower()

    m = re.search(r"patient[_\s]*id\s*[:=]?\s*(\d+)", q)
    if m:
        return m.group(1)

    m = re.search(r"patient\s+(\d+)", q)
    if m:
        return m.group(1)

    nums = re.findall(r"\b\d+\b", q)
    if len(nums) == 1:
        return nums[0]
    return None


# -------------------------------------------------------------------------
# LLM-based explanation & report summarisation tools
# -------------------------------------------------------------------------
def explanation_mode_answer(
    question: str,
    mode: str,
    report_text: str,
    probs: Dict[str, float],
) -> str:
    """Generate a clinician or student-mode explanation answer."""
    base_summary = "### AI Classification Summary\n"
    base_summary += (
        summarize_probs(probs) if probs else "No classification run yet."
    ) + "\n\n"
    base_summary += "### AI-generated CXR Report\n"
    base_summary += (report_text.strip() or "No report generated yet.") + "\n"

    if mode == "Clinician":
        return base_summary

    if not is_on_topic(question):
        return (
            "> 🧑‍🏫 This tutor only discusses the current chest X-ray and thoracic imaging. "
            "Please rephrase your question to focus on this study "
            "(for example: 'How do you distinguish pneumonia from atelectasis here?')."
        )

    if openai_client is None:
        return (
            "> ⚠️ Student mode is unavailable because the OpenAI client is not configured. "
            "Set OPENAI_API_KEY to enable GPT-4o-mini explanations."
        )

    system_prompt = (
        "You are a radiology tutor teaching a medical student about a single chest X-ray. "
        "You are given an AI-generated classification summary and report. "
        "Your goals:\n"
        "1. Explain the findings and differential in simple but precise terms.\n"
        "2. Emphasize visual patterns for pneumonia vs atelectasis vs edema vs effusion.\n"
        "3. Tie the explanation to THIS image only; avoid speculation beyond it.\n"
        "4. If the user asks unrelated topics, politely refuse and redirect.\n"
        "5. Do NOT give treatment recommendations.\n"
        "6. Explicitly state this is educational, not a clinical decision tool."
    )

    user_content = (
        "Classification summary:\n"
        + (summarize_probs(probs, top_k=7) if probs else "No classification available yet.")
        + "\n\nFull AI report:\n"
        + (report_text.strip() or "No report available yet.")
        + "\n\nStudent question:\n"
        + question.strip()
    )

    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=700,
    )

    tutor_answer = completion.choices[0].message.content
    return tutor_answer


def summarize_reports_for_patient_llm(
    patient_id: str,
    report_rows: List[Dict[str, str]],
) -> str:
    """Summarise multiple prior reports for a single patient using GPT-4o-mini."""
    if not report_rows:
        return "I could not find any saved reports for this patient in the database."

    if openai_client is None:
        lines = [f"### Report summaries for patient {patient_id} (no LLM – simple truncation)\n"]
        for row in report_rows:
            raw = (row.get("report_text") or "").replace("\n", " ").strip()
            if not raw:
                summary = "_No stored report text for this study._"
            else:
                snippet = raw
                if len(snippet) > 280:
                    snippet = snippet[:280] + "..."
                summary = snippet
            lines.append(
                f"- **Study {row.get('study_id', '')}** ({row.get('created_at', '')}): {summary}"
            )
        return "\n".join(lines)

    items = []
    for row in report_rows:
        report_text = (row.get("report_text") or "").strip()
        if len(report_text) > 1200:
            report_text = report_text[:1200] + "..."
        items.append(
            f"Study {row.get('study_id','')} (created_at={row.get('created_at','')}):\n{report_text}"
        )

    user_content = (
        f"You are given multiple chest X-ray reports for a single patient with internal id {patient_id}.\n"
        "For each report, produce a short 1–2 sentence summary in markdown bullet list form.\n"
        "Highlight key imaging findings and, if possible, note any evolution over time, "
        "but do not speculate beyond what is written in the reports.\n\n"
        "Reports:\n\n" + "\n\n".join(items)
    )

    system_prompt = (
        "You are a radiology assistant summarising prior chest X-ray reports for a clinician. "
        "Be concise, clinical, and avoid giving treatment recommendations."
    )

    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return completion.choices[0].message.content
