"""Patient registry database models and operations."""

from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from config import DATABASE_URL
from .qc import load_cxr_from_path
from .classifier import summarize_probs


engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Patient(Base):
    """SQLAlchemy model representing a patient."""

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(64), nullable=True, index=True)  # e.g. hospital MRN
    name = Column(String(128), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(16), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    studies = relationship("Study", back_populates="patient")


class Study(Base):
    """SQLAlchemy model representing a single CXR study for a patient."""

    __tablename__ = "studies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True, nullable=False)

    image_path = Column(String(512), nullable=True)
    image_hash = Column(String(64), index=True, nullable=True)

    qc_json = Column(Text, nullable=True)
    cnn_probs_json = Column(Text, nullable=True)
    report_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="studies")


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def compute_image_hash_from_path(image_path: str) -> str:
    """Compute a stable hash from the image pixels (256×256 grayscale)."""
    img, _ = load_cxr_from_path(image_path)
    img_resized = img.convert("L").resize((256, 256))
    data = img_resized.tobytes()
    return hashlib.sha256(data).hexdigest()


def extract_impression(report: str, max_chars: int = 200) -> str:
    """Extract an 'Impression' section or a short snippet from a report."""
    if not report:
        return ""
    lower = report.lower()
    idx = lower.find("impression")
    if idx != -1:
        snippet = report[idx:]
    else:
        snippet = report
    snippet = snippet.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return snippet


def build_history_rows(session, patient_id: int) -> List[Dict[str, str]]:
    """Build a list of dictionaries summarising all past studies for a patient."""
    studies = (
        session.query(Study)
        .filter(Study.patient_id == patient_id)
        .order_by(Study.created_at.asc())
        .all()
    )

    rows: List[Dict[str, str]] = []
    for st in studies:
        try:
            probs = json.loads(st.cnn_probs_json or "{}")
        except Exception:
            probs = {}
        top_probs = summarize_probs(probs, top_k=3) if probs else ""
        impression = extract_impression(st.report_text or "")
        rows.append(
            {
                "study_id": st.id,
                "created_at": st.created_at.isoformat() if st.created_at else "",
                "image_hash": (st.image_hash or "")[:12],
                "top_probs": top_probs,
                "impression": impression,
                "image_path": st.image_path or "",
            }
        )
    return rows


def save_study_to_db(
    patient_id_text: Optional[str],
    name: Optional[str],
    age: Optional[float],
    gender: Optional[str],
    image_path: Optional[str],
    qc_dict: Optional[Dict[str, Any]],
    probs_dict: Optional[Dict[str, float]],
    report_text: Optional[str],
):
    """Insert or append a study for an existing or new patient."""
    if image_path is None:
        raise ValueError("No image uploaded – cannot save study.")

    session = SessionLocal()
    try:
        patient = None
        if patient_id_text:
            try:
                pid_int = int(str(patient_id_text).strip())
                patient = session.get(Patient, pid_int)
            except ValueError:
                patient = (
                    session.query(Patient)
                    .filter(Patient.external_id == str(patient_id_text).strip())
                    .first()
                )

        if patient is None:
            patient = Patient(
                external_id=None,
                name=(name or None),
                age=int(age) if age not in (None, "") else None,
                gender=(gender or None),
            )
            session.add(patient)
            session.flush()

        image_hash = compute_image_hash_from_path(image_path)

        study = Study(
            patient_id=patient.id,
            image_path=image_path,
            image_hash=image_hash,
            qc_json=json.dumps(qc_dict or {}),
            cnn_probs_json=json.dumps(probs_dict or {}),
            report_text=report_text or "",
        )
        session.add(study)
        session.commit()

        history_rows = build_history_rows(session, patient.id)
        return str(patient.id), str(study.id), history_rows
    finally:
        session.close()


def retrieve_patient_history(
    patient_id_text: Optional[str],
    image_path: Optional[str],
):
    """Retrieve an existing patient and history via ID / external ID / image hash."""
    session = SessionLocal()
    try:
        patient = None
        matched_on = "none"

        if patient_id_text:
            try:
                pid_int = int(str(patient_id_text).strip())
                patient = session.get(Patient, pid_int)
                if patient is not None:
                    matched_on = f"patient_id={pid_int}"
            except ValueError:
                pid_ext = str(patient_id_text).strip()
                patient = (
                    session.query(Patient)
                    .filter(Patient.external_id == pid_ext)
                    .first()
                )
                if patient is not None:
                    matched_on = f"external_id={pid_ext}"

        if patient is None and image_path is not None:
            image_hash = compute_image_hash_from_path(image_path)
            study = (
                session.query(Study)
                .filter(Study.image_hash == image_hash)
                .order_by(Study.created_at.desc())
                .first()
            )
            if study is not None:
                patient = study.patient
                matched_on = "uploaded CXR hash"

        if patient is None:
            return None, False, "no match", []

        history_rows = build_history_rows(session, patient.id)
        return patient, False, matched_on, history_rows
    finally:
        session.close()


def fetch_patient_reports_for_patient_id(patient_id_text: Optional[str]):
    """Fetch all studies + raw reports for a given patient_id or external_id."""
    session = SessionLocal()
    try:
        if not patient_id_text or str(patient_id_text).strip() == "":
            return None, []

        patient = None
        try:
            pid_int = int(str(patient_id_text).strip())
            patient = session.get(Patient, pid_int)
        except ValueError:
            pid_ext = str(patient_id_text).strip()
            patient = (
                session.query(Patient)
                .filter(Patient.external_id == pid_ext)
                .first()
            )

        if patient is None:
            return None, []

        studies = (
            session.query(Study)
            .filter(Study.patient_id == patient.id)
            .order_by(Study.created_at.asc())
            .all()
        )

        rows = []
        for st in studies:
            rows.append(
                {
                    "study_id": st.id,
                    "created_at": st.created_at.isoformat() if st.created_at else "",
                    "report_text": (st.report_text or "").strip(),
                }
            )
        return patient, rows
    finally:
        session.close()


# Initialise DB schema on import
init_db()
