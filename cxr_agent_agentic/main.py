"""Entry point for running the CXR-Agent Gradio app.

Usage (in Colab):

```python
%cd /content
# (clone / upload this cxr_agent folder under /content)
%run cxr_agent/main.py
```

Make sure you have:
  - Mounted Google Drive (if using Drive paths) via `from google.colab import drive; drive.mount('/content/drive')`
  - Installed the required libraries (transformers, timm, gradio, sqlalchemy, etc.)
  - Set `OPENAI_API_KEY` if you want GPT-4o-mini based routing and explanations.
"""

from __future__ import annotations

import os

import gradio as gr

from ui import build_interface


def main():
    """Launch the Gradio interface."""
    demo = build_interface()
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
