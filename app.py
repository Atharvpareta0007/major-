import os
import tempfile
from typing import Tuple, Dict

import gradio as gr

from inference_utils import predict_from_audio_path


def predict_interface(audio: Tuple[int, str]) -> Tuple[str, dict, str]:
    """Gradio interface function.

    `audio` is a (sample_rate, file_path) tuple when using gr.Audio with type="filepath".
    Returns: (pred_label, probs_dict, model_type)
    """
    if audio is None:
        return "No audio provided", {}, ""

    if isinstance(audio, tuple) and len(audio) == 2:
        _, file_path = audio
    else:
        file_path = audio

    pred, probs, model_type = predict_from_audio_path(file_path)
    return pred, probs, model_type


with gr.Blocks(title="Speech Screening - Depression Classifier") as demo:
    gr.Markdown("""
    # Speech Screening - Depression Classifier

    Record or upload a short speech sample (e.g., 5-15 seconds). The model will predict whether the sample is **MDD** or **HC**.

    Note: This is a research prototype for demonstration and not a medical diagnostic tool.
    """)

    with gr.Row():
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speech Input")
    with gr.Row():
        btn = gr.Button("Predict")

    with gr.Row():
        pred = gr.Textbox(label="Predicted Class", interactive=False)
        model_type = gr.Textbox(label="Model Used", interactive=False)

    probs = gr.JSON(label="Class Probabilities")

    btn.click(fn=predict_interface, inputs=audio, outputs=[pred, probs, model_type])


if __name__ == "__main__":
    # By default Gradio runs on http://127.0.0.1:7860
    demo.launch()



