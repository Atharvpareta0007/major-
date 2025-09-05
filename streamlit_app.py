import os
import io
import tempfile
from typing import Dict, Tuple, Optional

import numpy as np
import plotly.express as px
import streamlit as st


# ----------- Theme & Styling Helpers -----------
def inject_global_css(dark_mode: bool) -> None:
    """Inject custom CSS for dark/light theme and subtle animations."""
    # Soothing palette
    primary = "#6C63FF"  # purple
    secondary = "#00BFA6"  # teal
    accent = "#4FC3F7"  # blue

    bg_light = "#F6F8FB"
    text_light = "#111827"
    card_light = "#FFFFFF"

    bg_dark = "#0E1117"
    text_dark = "#E5E7EB"
    card_dark = "#1F2937"

    bg = bg_dark if dark_mode else bg_light
    text = text_dark if dark_mode else text_light
    card = card_dark if dark_mode else card_light

    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {primary};
            --secondary: {secondary};
            --accent: {accent};
            --bg: {bg};
            --text: {text};
            --card: {card};
        }}

        .stApp {{
            background-color: var(--bg);
            color: var(--text);
        }}

        .card {{
            background: var(--card);
            padding: 1.25rem 1rem;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            transition: transform 200ms ease, box-shadow 200ms ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        }}

        .accent-title {{
            color: var(--primary);
        }}

        .fade-in {{
            animation: fadeIn 600ms ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        .footer {{
            color: var(--text);
            opacity: 0.8;
            font-size: 0.9rem;
            padding: 1rem 0;
            border-top: 1px solid rgba(125,125,125,0.15);
            text-align: center;
            margin-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------- Mock/Real Prediction Wrappers -----------
def _attempt_real_predict(audio_path: str) -> Optional[Tuple[str, Dict[str, float], str]]:
    try:
        from inference_utils import predict_from_audio_path
        return predict_from_audio_path(audio_path)
    except Exception:
        return None


def _random_probs(classes: Tuple[str, ...] = ("HC", "MDD")) -> Dict[str, float]:
    p = np.random.dirichlet(np.ones(len(classes)))
    return {cls: float(prob) for cls, prob in zip(classes, p)}


def predict_audio(audio_file) -> Tuple[str, Dict[str, float], str]:
    """Predict from uploaded audio file (BytesIO or tempfile). Falls back to random."""
    # Write to temp and try real model
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp.flush()
            audio_path = tmp.name
        out = _attempt_real_predict(audio_path)
        if out is not None:
            return out
    except Exception:
        pass
    # Fallback mock
    probs = _random_probs()
    pred = max(probs, key=probs.get)
    return pred, probs, "student (mock)"


def predict_mic(audio_bytes: bytes) -> Tuple[str, Dict[str, float], str]:
    """Predict from microphone audio bytes. Falls back to random."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio_path = tmp.name
        out = _attempt_real_predict(audio_path)
        if out is not None:
            return out
    except Exception:
        pass
    probs = _random_probs()
    pred = max(probs, key=probs.get)
    return pred, probs, "student (mock)"


def predict_eeg(eeg_file) -> Tuple[str, Dict[str, float], str]:
    """Placeholder EEG predictor. Replace with real EEG inference if needed."""
    # TODO: parse EEG CSV and run teacher model when available
    probs = _random_probs()
    pred = max(probs, key=probs.get)
    return pred, probs, "teacher (mock)"


# ----------- UI Components -----------
def page_home(dark_mode: bool) -> None:
    st.markdown(
        "<h1 class='fade-in'>🧠 AI-Powered Depression Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        A research demo using cross-modal knowledge distillation to screen for depression from speech and EEG signals.
        For demonstration only — not a medical diagnostic tool.
        """
    )
    # Banner image (replace with local file if preferred)
    st.image(
        "https://images.unsplash.com/photo-1527236438218-d82077ae1f85?q=80&w=1600&auto=format&fit=crop",
        caption="Neural patterns and mental health",
        use_column_width=True,
    )


def page_about(dark_mode: bool) -> None:
    st.subheader("About the Project")
    st.markdown(
        """
        - 🧩 <b>Teacher Model</b>: Fuses EEG + Audio. Provides rich supervision.
        - 🎙️ <b>Student Model</b>: Audio-only, distilled from the teacher for lightweight deployment.
        - 🔄 <b>Knowledge Distillation</b>: Transfers knowledge from teacher to student.
        """,
        unsafe_allow_html=True,
    )

    # Flow diagram placeholder image
    st.image(
        "https://images.unsplash.com/photo-1526378722484-bd91ca387e72?q=80&w=1600&auto=format&fit=crop",
        caption="Multimodal → Knowledge Distillation → Lightweight Inference",
        use_column_width=True,
    )

    st.markdown("### Datasets")
    st.markdown(
        """
        - 📁 <b>Audio</b>: Lanzhou 2015 audio dataset (MDD / HC) per subject.
        - 🧪 <b>EEG</b>: EEG_128channels_resting_lanzhou_2015 (resting-state EEG recordings).
        - 🔧 Feature extraction with MFCCs, Chroma, Mel-spectrogram, spectral features.
        """,
        unsafe_allow_html=True,
    )


def _viz_probs(probs: Dict[str, float], chart_type: str = "bar") -> None:
    labels = list(probs.keys())
    values = [probs[k] for k in labels]
    if chart_type == "pie":
        fig = px.pie(values=values, names=labels, title="Confidence")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.bar(x=labels, y=values, title="Confidence", color=labels,
                     color_discrete_sequence=["#00BFA6", "#6C63FF"])  # teal, purple
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


def page_detection(dark_mode: bool) -> None:
    st.subheader("Detection")
    st.caption("Choose input type and run detection. Models are mocked until you add trained weights.")

    col1, col2 = st.columns([1, 1])
    with col1:
        input_mode = st.radio("Input Mode", ["Upload Audio", "Record Audio", "Upload EEG (optional)"])
        chart_type = st.selectbox("Confidence Visualization", ["bar", "pie"], index=0)
    with col2:
        model_choice = st.selectbox("Model", ["Student (Audio)", "Teacher (EEG+Audio)"], index=0)

    audio_file = None
    mic_bytes: Optional[bytes] = None
    eeg_file = None

    if input_mode == "Upload Audio":
        audio_file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3", "flac"])
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
    elif input_mode == "Record Audio":
        try:
            from streamlit_mic_recorder import mic_recorder
            st.info("Use the button below to start/stop recording.")
            audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", key="rec")
            if audio and isinstance(audio, dict) and audio.get("bytes"):
                mic_bytes = audio["bytes"]
                st.audio(io.BytesIO(mic_bytes), format="audio/wav")
        except Exception:
            st.warning("Microphone component unavailable. Please install 'streamlit_mic_recorder' or use upload.")
    else:
        eeg_file = st.file_uploader("Upload EEG CSV (optional for teacher)", type=["csv"]) 

    run = st.button("Run Detection", use_container_width=True, type="primary")

    if run:
        with st.spinner("Running detection..."):
            if model_choice.startswith("Student"):
                if audio_file is not None:
                    pred, probs, model_used = predict_audio(audio_file)
                elif mic_bytes is not None:
                    pred, probs, model_used = predict_mic(mic_bytes)
                else:
                    st.error("Please provide audio input.")
                    return
            else:
                # Teacher path (mock) — if both EEG and audio are available, could combine later
                if eeg_file is not None:
                    pred, probs, model_used = predict_eeg(eeg_file)
                elif audio_file is not None:
                    pred, probs, model_used = predict_audio(audio_file)
                elif mic_bytes is not None:
                    pred, probs, model_used = predict_mic(mic_bytes)
                else:
                    st.error("Please provide EEG CSV and/or audio input.")
                    return

        label_readable = "Depressed" if pred.upper() == "MDD" else "Not Depressed"
        st.success(f"Prediction: {label_readable} ({pred}) | Model: {model_used}")
        _viz_probs(probs, chart_type=chart_type)

    st.markdown("<div class='footer'>Developed by Atharv Pareta – Final Year Project</div>", unsafe_allow_html=True)


# ----------- Main App -----------
def main() -> None:
    st.set_page_config(
        page_title="Depression Detection (EEG + Audio)",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar with navigation and theme toggle
    with st.sidebar:
        st.image(
            "https://images.unsplash.com/photo-1526250599556-4ae3b36f7a9a?q=80&w=800&auto=format&fit=crop",
            caption="Mental Health",
            use_column_width=True,
        )
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Home", "About Project", "Detection"], index=0)

        dark_mode = st.toggle("Dark Mode", value=True)
        st.caption("Toggle for soothing dark/light themes")

    inject_global_css(dark_mode)

    if page == "Home":
        page_home(dark_mode)
    elif page == "About Project":
        page_about(dark_mode)
    else:
        page_detection(dark_mode)


if __name__ == "__main__":
    main()


