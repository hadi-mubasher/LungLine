"""MedGemma-based chest X-ray report generation."""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText

from config import MEDGEMMA_BASE_ID, MEDGEMMA_PEFT_DIR, HUGGING_FACE_API_KEY_ENV


_MEDGEMMA_MODEL = None
_MEDGEMMA_PROCESSOR = None


def load_medgemma():
    """Load the MedGemma base model + PEFT adapter once and cache them."""
    import os
    global _MEDGEMMA_MODEL, _MEDGEMMA_PROCESSOR
    hf_token=os.environ.get(HUGGING_FACE_API_KEY_ENV)
    if _MEDGEMMA_MODEL is None or _MEDGEMMA_PROCESSOR is None:
        base = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_BASE_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        _MEDGEMMA_MODEL = PeftModel.from_pretrained(base, MEDGEMMA_PEFT_DIR, token=hf_token)
        _MEDGEMMA_MODEL.eval()
        _MEDGEMMA_PROCESSOR = AutoProcessor.from_pretrained(MEDGEMMA_BASE_ID)
        print("MedGemma model + PEFT adapter loaded.")
    return _MEDGEMMA_MODEL, _MEDGEMMA_PROCESSOR


def generate_cxr_report(image: Image.Image) -> str:
    """Use MedGemma to generate a structured CXR report for a single image."""
    model, processor = load_medgemma()
    image_rgb = image.convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert chest radiologist. "
                        "Write a clear chest X-ray report with sections: Indication, Technique, Findings, Impression. "
                        "Mention uncertainties explicitly and do not give management recommendations."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """You are a CXR analyst. Return only chest X-ray FINDINGS in 1–3 short sentences. No headings, no patient/date/indication,
    no “preliminary”, no disclaimers. Do not assume sex/age/view beyond provided text. Use precise radiology wording. If normal,
    write a single normal line (e.g., “Lungs clear. Cardiomediastinal silhouette normal. No pleural effusion or pneumothorax.”).
    If abnormal, state the key positives only, concise (e.g., “Feeding tube projects to upper abdomen; bibasilar atelectasis; no pneumothorax or pleural effusion.”).
    Start directly with the findings."""},
                {"type": "image", "image": image_rgb},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
        )
        generation = generation[0, input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded.strip()