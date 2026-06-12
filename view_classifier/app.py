"""Gradio demo: upload a vehicle photo → pose bin + azimuth + confidence.

Run from repo root:

  PYTHONPATH=. .venv/bin/python -m view_classifier.app
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from view_classifier.labels import POSE_CLASSES
from view_classifier.predict import (
    DEFAULT_CKPT,
    DEMO_IMAGE_DIR,
    format_prediction,
    list_demo_images,
    predict_image,
)


def _prob_chart(probs: dict) -> plt.Figure:
    names = list(POSE_CLASSES)
    vals = [probs[n] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#1b7f3a" if v == max(vals) else "#8a8a8a" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title("Pose class probabilities")
    fig.tight_layout()
    return fig


def run_predict(image: Image.Image | None, ckpt: str, device: str):
    if image is None:
        raise gr.Error("Upload or select an image first.")
    result = predict_image(image, ckpt=ckpt, device=device)
    return format_prediction(result), _prob_chart(result["probs"]), result


def build_app(ckpt: Path, device: str) -> gr.Blocks:
    examples = [str(p) for p in list_demo_images()]
    with gr.Blocks(title="Vehicle View Classifier") as demo:
        gr.Markdown(
            """
# Vehicle View Classifier

Upload a car photo (or pick a sample). The model predicts:

- **8-way view** — front / corners / sides / rear  
- **azimuth angle** — 0° = front, clockwise, 90° = right  
- **confidence** — low confidence often means close-up, document, or extreme wreck  

Personal research demo — not a product.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Photo", height=360)
                ckpt_box = gr.Textbox(value=str(ckpt), label="Checkpoint path")
                device_box = gr.Dropdown(
                    choices=["auto", "mps", "cuda", "cpu"],
                    value=device,
                    label="Device",
                )
                btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Markdown(label="Prediction")
                chart = gr.Plot(label="Class probabilities")
                raw = gr.JSON(label="Raw output")

        if examples:
            gr.Examples(
                examples=examples,
                inputs=image,
                label=f"Sample damage photos ({DEMO_IMAGE_DIR.name})",
            )

        btn.click(run_predict, inputs=[image, ckpt_box, device_box], outputs=[summary, chart, raw])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio public link (temporary)")
    args = parser.parse_args()

    app = build_app(args.ckpt, args.device)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
