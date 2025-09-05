import os
import pickle
from functools import lru_cache
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Reuse feature extraction from teacher module
from teacher_model import extract_features


# Device for inference
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_preprocessing_objects(pkl_path: str = 'preprocessing_objects.pkl') -> Tuple[object, object, int, int]:
    """Load scaler, label encoder, input_size, num_classes from pickle file.

    The pickle should contain keys: 'scaler', 'label_encoder', 'input_size', 'num_classes'.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Preprocessing pickle not found: {pkl_path}. Please train teacher_model.py first."
        )
    with open(pkl_path, 'rb') as f:
        data: Dict[str, object] = pickle.load(f)
    scaler = data['scaler']
    label_encoder = data['label_encoder']
    input_size = int(data['input_size'])
    num_classes = int(data['num_classes'])
    return scaler, label_encoder, input_size, num_classes


def _vectorize_features(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Create the same feature vector layout used during training."""
    return np.concatenate([
        features['mfccs'],
        features['chroma'],
        features['mel_spec'],
        [features['zcr']],
        [features['spectral_centroid']],
        [features['spectral_rolloff']],
        [features['spectral_bandwidth']],
    ])


@lru_cache(maxsize=1)
def load_model_for_inference(
    weights_student: str = 'student_model.pth',
    weights_teacher: str = 'teacher_model.pth',
) -> Tuple[torch.nn.Module, str]:
    """Load student if available, else teacher model for inference.

    Returns (model, model_type) with model on correct device.
    """
    # Load preprocessing to know input_size/num_classes
    _, _, input_size, num_classes = load_preprocessing_objects()

    model: Optional[torch.nn.Module] = None
    model_type = 'student'

    if os.path.exists(weights_student):
        try:
            from student_model import create_student_model
            model = create_student_model(input_size, num_classes)
            model.load_state_dict(torch.load(weights_student, map_location=device))
            model.eval()
            return model, model_type
        except Exception:
            model = None

    # Fallback to teacher
    model_type = 'teacher'
    from teacher_model import create_model as create_teacher_model
    model = create_teacher_model(input_size, num_classes)
    if os.path.exists(weights_teacher):
        model.load_state_dict(torch.load(weights_teacher, map_location=device))
    model.eval()
    return model, model_type


def predict_from_audio_path(
    audio_path: str,
    pkl_path: str = 'preprocessing_objects.pkl',
) -> Tuple[str, Dict[str, float], str]:
    """Predict class from an audio file path.

    Returns: (predicted_label, probabilities_dict, model_type)
    """
    scaler, label_encoder, input_size, _ = load_preprocessing_objects(pkl_path)

    feats = extract_features(audio_path)
    if feats is None:
        raise ValueError(f"Failed to extract features from audio: {audio_path}")
    vec = _vectorize_features(feats)

    if vec.shape[0] != input_size:
        raise ValueError(f"Feature size mismatch: got {vec.shape[0]}, expected {input_size}")

    vec_norm = scaler.transform(vec.reshape(1, -1))

    model, model_type = load_model_for_inference()
    with torch.no_grad():
        x = torch.tensor(vec_norm, dtype=torch.float32, device=device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    classes = list(label_encoder.classes_)
    probs_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    return pred_label, probs_dict, model_type



