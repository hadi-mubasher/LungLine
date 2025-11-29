"""Gradio UI wiring for the CXR-Agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gradio as gr

from orchestrator import (
    chat_handler,
    insert_to_registry,
    retrieve_from_registry,
    on_history_select,
)
from modules.qc import load_cxr_from_path, save_display_image


def reset_state():
    """Return default values for state variables."""
    return [], {}, {}, "", [], []


def on_image_upload(image):
    """Handle new image upload, convert/save for display, and reset state."""
    if image is None:
        return (
            None,
            [],
            {},
            {},
            "",
            [],
            [],
            "",
            [],
            "",
            "",
            "",
            [],
            [],
        )

    tmp_path = image  # Gradio file path
    pil_img, _ = load_cxr_from_path(tmp_path)
    display_path = save_display_image(pil_img, tmp_path)

    (
        chat_history,
        qc_dict,
        probs_dict,
        report_text,
        heatmap_gallery,
        history_rows,
    ) = reset_state()

    return (
        display_path,
        chat_history,
        qc_dict,
        probs_dict,
        report_text,
        heatmap_gallery,
        history_rows,
        "",
        [],
        "",
        "",
        "",
        [],
        [],
    )


def build_interface() -> gr.Blocks:
    """Create the full Gradio Blocks interface."""
    LIGHT_CSS = """
    .gradio-container {
        background: radial-gradient(circle at top, #f5fbff 0, #f9fafb 40%, #ffffff 100%);
        color: #111827;
    }

    /* Make cards light grey */
    .section-card {
        background-color: #7393B3;
        border-radius: 18px;
        padding: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.04);
    }

    .section-card .gradio-markdown h3,
    .section-card .gradio-markdown h4 {
        margin-top: 0.25rem;
        color: #B2BEB5;
    }

    /* Darken general text */
    .gradio-container, 
    .gradio-container * {
        color: #E5E4E2;
    }

    label, .small-label {
        color: #E5E4E2 !important;
    }

    .top-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color:  #111827;
        margin-bottom: 0.25rem;
    }

    .top-header p {
        color: #111827;
        font-size: 0.95rem;
    }

    .top-header ul {
        color: #E5E4E2;
        font-size: 0.95rem;
    }

    .small-label {
        color: #E5E4E2,
        font-size: 0.85rem;
    }

    /* Align send button row to the right */
    .send-row {
        display: flex;
        justify-content: flex-end;
        margin-top: 0.25rem;
    }

    .send-row button {
        min-width: 120px;
    }
    """

    with gr.Blocks(
        title="CXR-Agent (Agentic Tool Routing)",
        theme=gr.themes.Soft(
            primary_hue="sky",
            secondary_hue="teal",
            neutral_hue="gray",
        ),
        css=LIGHT_CSS,
    ) as demo:
        # Top header
        gr.Markdown(
            """
<div class="top-header">
  <h1>🫁 LungLine: Agentic Chest X-ray Assistant</h1>
  <p>
    Upload a chest X-ray, then chat with the agent to obtain reports, heatmaps, and teaching explanations.</p>
    <p>
    You can also save and revisit cases via the patient registry.</p>
    <p>1. Ask for a diagnosis or structured report.</p>
    <p>2. Request heatmaps for localisation.</p>
    <p>3. Switch explanation level between Clinician and Student.</p>
    <p>4. Use the patient registry to save and retrieve studies.
  </p>


</div>
""",
        )

        with gr.Row():
            # LEFT COLUMN: Image + QC
            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("### 📷 Current Study", elem_classes=["small-label"])

                image_input = gr.File(
                    label="Upload chest X-ray (PNG / JPEG / DICOM)",
                    file_types=[".png", ".jpg", ".jpeg", ".dcm", ".dicom"],
                )

                image_display = gr.Image(
                    label="Current CXR",
                    interactive=False,
                )

                explanation_mode = gr.Radio(
                    choices=["Clinician", "Student"],
                    value="Clinician",
                    label="Explanation mode",
                )

                with gr.Accordion("Quality Control & Classifier Output", open=False):
                    qc_panel = gr.JSON(label="QC result")

                    probs_table = gr.Dataframe(
                        headers=["Label", "Probability"],
                        datatype=["str", "number"],
                        label="Classifier probabilities",
                    )

            # MIDDLE COLUMN: Chat + report + heatmaps
            with gr.Column(scale=2, elem_classes=["section-card"]):
                gr.Markdown("### 💬 CXR-Agent Chat")

                chatbot = gr.Chatbot(
                    [],
                    label="Dialogue",
                    type="messages",
                )

                # Question box full-width
                user_message = gr.Textbox(
                    label="Your question / instruction",
                    placeholder=(
                        "e.g., 'Generate a report', 'Show me heatmaps', "
                        "'Explain this case to a student level', "
                        "'Summarise prior reports for this patient'"
                    ),
                    lines=2,
                )

                # Send button right-aligned beneath textbox
                with gr.Row(elem_classes=["send-row"]):
                    send_btn = gr.Button("Send", variant="primary")

                with gr.Accordion("AI-Generated Report", open=True):
                    report_box = gr.Textbox(
                        label="Report",
                        lines=10,
                    )

                with gr.Accordion("Heatmaps", open=False):
                    heatmap_gallery = gr.Gallery(
                        label="Heatmaps (top-3 localisation labels)",
                        columns=3,
                        rows=1,
                        show_label=True,
                    )
                    heatmap_status = gr.Markdown("No heatmaps generated yet.")

            # RIGHT COLUMN: Registry
            with gr.Column(scale=1, elem_classes=["section-card"]):
                gr.Markdown("### 🗂️ Patient Registry")

                patient_id_text = gr.Textbox(label="Patient ID (optional)")
                patient_name = gr.Textbox(label="Name (optional)")
                patient_age = gr.Slider(
                                  minimum=1,
                                  maximum=99,
                                  step=1,
                                  value=30,           # default age, change if you like
                                  label="Age"
                              )
                patient_gender = gr.Dropdown(
                              choices=["Male", "Female"],
                              label="Gender",
                              value="Male"        # optional default
                          )

                registry_status = gr.Markdown("Ready.")

                with gr.Row():
                    save_btn = gr.Button("💾 Save current study to DB")
                    retrieve_btn = gr.Button("🔍 Retrieve patient history")

                studies_table = gr.Dataframe(
                    headers=[
                        "study_id",
                        "created_at",
                        "image_hash",
                        "top_probs",
                        "impression",
                    ],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Patient history",
                )

                history_rows_state = gr.State([])

                history_image = gr.Image(
                    label="Selected past CXR",
                    interactive=False,
                )

            # Hidden / state components
            chat_history_state = gr.State([])
            qc_dict_state = gr.State({})
            probs_dict_state = gr.State({})
            report_text_state = gr.State("")
            heatmap_gallery_state = gr.State([])
            heatmap_status_state = gr.State("")
            patient_id_state = gr.State("")

            # Image upload → reset and show image
            image_input.change(
                fn=on_image_upload,
                inputs=[image_input],
                outputs=[
                    image_display,
                    chat_history_state,
                    qc_dict_state,
                    probs_dict_state,
                    report_text_state,
                    heatmap_gallery_state,
                    history_rows_state,
                    registry_status,
                    studies_table,
                    heatmap_status_state,
                    patient_id_state,
                    patient_name,
                    patient_age,
                    patient_gender,
                ],
            )

            # Chat / send message
            send_btn.click(
                fn=chat_handler,
                inputs=[
                    user_message,
                    chat_history_state,
                    explanation_mode,
                    image_input,
                    qc_dict_state,
                    probs_dict_state,
                    report_text_state,
                    heatmap_gallery_state,
                    heatmap_status_state,
                    patient_id_state,
                ],
                outputs=[
                    chat_history_state,
                    explanation_mode,
                    qc_dict_state,
                    probs_dict_state,
                    report_text_state,
                    qc_panel,
                    probs_table,
                    report_box,
                    heatmap_gallery,
                    heatmap_status,
                ],
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=[user_message],
            ).then(
                fn=lambda h: h,
                inputs=[chat_history_state],
                outputs=[chatbot],
            )

            # Save to registry
            save_btn.click(
                fn=insert_to_registry,
                inputs=[
                    patient_id_text,
                    patient_name,
                    patient_age,
                    patient_gender,
                    image_input,
                    qc_dict_state,
                    probs_dict_state,
                    report_text_state,
                ],
                outputs=[
                    registry_status,
                    patient_id_state,
                    gr.Textbox(visible=False),
                    studies_table,
                    history_rows_state,
                    history_image,
                ],
            )

            # Retrieve from registry
            retrieve_btn.click(
                fn=retrieve_from_registry,
                inputs=[
                    patient_id_text,
                    image_input,
                ],
                outputs=[
                    registry_status,
                    patient_id_state,
                    gr.Textbox(visible=False),
                    studies_table,
                    history_rows_state,
                    history_image,
                ],
            )

            # When a row is selected in the history table, show the image
            studies_table.select(
                fn=on_history_select,
                inputs=[studies_table, history_rows_state],
                outputs=[history_image],
            )

    return demo
