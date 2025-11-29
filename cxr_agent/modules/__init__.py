"""Subpackage containing all functional modules for the CXR-Agent.

Modules:
  - qc.py          : image loading + QC using CLIP + heuristics
  - classifier.py  : 14-label SwinV2 classifier
  - report.py      : MedGemma report generation
  - heatmaps.py    : SwinV2-based localization heatmaps
  - db.py          : SQLAlchemy models + DB utilities
  - agent_tools.py : agentic tool schema + GPT-4o routing helpers
"""
