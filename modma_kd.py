"""
Depression Detection with Cross-Modal Knowledge Distillation (MODMA-ready skeleton)

This module provides:
- DepressionDataset: placeholder-ready dataset class for MODMA-like multi-modal data
- TeacherModel: speech + EEG encoders with Improved Feature Fusion Block (IFFB) and classifier
- StudentModel: speech-only encoder and classifier
- Knowledge distillation loss: kd_total_loss
- Training utilities: train_teacher_model, train_student_model_with_kd
- Evaluation utility: evaluate_model
- Main scaffold showing the end-to-end flow

Notes:
- Data-specific loading is intentionally left as clear placeholders to be filled with
  MODMA file paths and actual preprocessing.
- Assumed shapes are documented near each component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import os

import numpy as np
import librosa
from scipy import io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# =============================
# Absolute dataset path bindings
# =============================
# Link to your local MODMA-like datasets (speech and EEG)
SPEECH_DATASET_DIR = "/Users/atharvpareta/Desktop/major/audio_lanzhou_2015"
EEG_DATASET_DIR = "/Users/atharvpareta/Desktop/major/EEG_128channels_resting_lanzhou_2015"


# =============================
# Dataset Preparation (MODMA)
# =============================


@dataclass
class DatasetConfig:
    """Configuration for dataset structure and preprocessing.

    Attributes:
        root_dir: Root directory containing MODMA-like data.
        speech_dirname: Subdirectory name for speech/audio files per subject.
        eeg_dirname: Subdirectory name for EEG files per subject.
        labels_filename: File name (or relative path) for labels if centralized.
        sample_rate: Expected audio sample rate for consistency.
        eeg_num_channels: Number of EEG channels to select (e.g., 64 for MODMA).
        speech_feature: One of {"mfcc", "logmel", "wav2vec2"}; controls placeholder expectations.
        window_seconds: EEG window length in seconds.
        window_stride_seconds: EEG sliding stride in seconds.
        speech_normalize_per_speaker: Whether to perform per-speaker normalization.
        eeg_normalize_per_channel: Whether to perform per-channel normalization.
    """

    root_dir: str
    speech_dirname: str = "speech"
    eeg_dirname: str = "eeg"
    labels_filename: Optional[str] = None
    sample_rate: int = 16000
    eeg_num_channels: int = 64
    speech_feature: str = "logmel"
    window_seconds: float = 5.0
    window_stride_seconds: float = 1.0
    speech_normalize_per_speaker: bool = True
    eeg_normalize_per_channel: bool = True


class DepressionDataset(Dataset):
    """Placeholder-ready dataset class for MODMA.

    Expected folder layout (conceptual):
        root_dir/
          subject_001/
            speech/  # raw audio files or precomputed features
            eeg/     # raw EEG files or precomputed signals
            label.txt or labels/...
          subject_002/
            ...

    __getitem__ returns a tuple:
        (speech_features: Tensor, eeg_features: Tensor, label: Tensor)

    Assumed shapes:
        - speech_features: (num_features, time_steps) or (time_steps, num_features)
          We will format as (channels, time) for CNN1d compatibility: (feat_dim, time)
        - eeg_features: (num_channels=64, time_steps) or windowed representation
          We will format as (channels, time)
        - label: scalar {0, 1}
    """

    def __init__(self, dataset_config: DatasetConfig, split: str = "train") -> None:
        super().__init__()
        self.config = dataset_config
        self.split = split

        # Discover subjects and build index. In practice, replace with real discovery of MODMA files.
        self.subject_ids: List[str] = self._discover_subjects()
        self.samples_index: List[Tuple[str, str]] = self._align_modalities()

    def _discover_subjects(self) -> List[str]:
        # Placeholder discovery: list dirs in root_dir as subject IDs
        if not os.path.isdir(self.config.root_dir):
            return []
        subject_ids = [d for d in os.listdir(self.config.root_dir) if os.path.isdir(os.path.join(self.config.root_dir, d))]
        subject_ids.sort()
        return subject_ids

    def _align_modalities(self) -> List[Tuple[str, str]]:
        """Create an index aligning speech and EEG segments.

        Returns list of tuples (subject_id, segment_id).
        Segmenting and alignment are placeholders; adapt to MODMA annotations.
        """
        samples: List[Tuple[str, str]] = []
        for subject_id in self.subject_ids:
            # Placeholder: pretend each subject has N segments
            num_segments = 10
            for segment_idx in range(num_segments):
                samples.append((subject_id, f"seg_{segment_idx:03d}"))
        return samples

    def __len__(self) -> int:
        return len(self.samples_index)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        subject_id, segment_id = self.samples_index[index]

        # Load raw data placeholders
        raw_audio = self._load_raw_audio(subject_id, segment_id)
        raw_eeg = self._load_raw_eeg(subject_id, segment_id)
        label_int = self._load_label(subject_id)

        # Preprocess
        speech_features = self._preprocess_speech(raw_audio, subject_id)
        eeg_features = self._preprocess_eeg(raw_eeg)

        # Convert types
        speech_tensor = torch.tensor(speech_features, dtype=torch.float32)
        eeg_tensor = torch.tensor(eeg_features, dtype=torch.float32)
        label_tensor = torch.tensor(label_int, dtype=torch.float32)

        return speech_tensor, eeg_tensor, label_tensor

    # -------------------------
    # Data loading placeholders
    # -------------------------
    def _load_raw_audio(self, subject_id: str, segment_id: str) -> np.ndarray:
        """Load raw audio waveform for a subject/segment from Lanzhou 2015 speech.

        Expects folder: SPEECH_DATASET_DIR/<subject_id>/ with wav files like 01.wav, 02.wav ...
        Selects a file deterministically based on segment_id, loads and center-crops/pads to window_seconds.
        Returns: np.ndarray of shape (num_samples,)
        """
        subject_path = os.path.join(SPEECH_DATASET_DIR, subject_id)
        if not os.path.isdir(subject_path):
            # Fallback to silence if subject folder missing
            return np.zeros(int(self.config.sample_rate * self.config.window_seconds), dtype=np.float32)

        wav_files = [f for f in sorted(os.listdir(subject_path)) if f.lower().endswith('.wav')]
        if len(wav_files) == 0:
            return np.zeros(int(self.config.sample_rate * self.config.window_seconds), dtype=np.float32)

        # Map segment_id like "seg_003" to an index
        try:
            seg_idx = int(segment_id.split('_')[-1])
        except Exception:
            seg_idx = 0
        file_path = os.path.join(subject_path, wav_files[seg_idx % len(wav_files)])

        # Load audio
        try:
            y, sr = librosa.load(file_path, sr=self.config.sample_rate, mono=True)
        except Exception:
            return np.zeros(int(self.config.sample_rate * self.config.window_seconds), dtype=np.float32)

        target_len = int(self.config.sample_rate * self.config.window_seconds)
        if len(y) == target_len:
            return y.astype(np.float32)
        if len(y) > target_len:
            # center-crop
            start = max(0, (len(y) - target_len) // 2)
            return y[start:start + target_len].astype(np.float32)
        # pad
        pad = target_len - len(y)
        return np.pad(y, (0, pad), mode='constant').astype(np.float32)

    def _load_raw_eeg(self, subject_id: str, segment_id: str) -> np.ndarray:
        """Load EEG for a subject/segment from Lanzhou EEG .mat files when possible.

        Strategy:
          - Search recursively in EEG_DATASET_DIR for a .mat file containing the subject_id digits.
          - Load a common variable key among {"data", "EEG", "eeg", "signal", "signals"}.
          - Ensure shape (channels, time). If needed, transpose from (time, channels).
          - Extract a centered window of length ~window_seconds at 200 Hz; fallback to 200 Hz if unknown.
          - If no file/key found, return deterministic noise for reproducibility.
        """
        # Default EEG sampling rate assumption
        assumed_eeg_fs = 200
        duration_seconds = self.config.window_seconds
        target_time = int(assumed_eeg_fs * duration_seconds)

        # Helper: deterministic noise fallback
        def _fallback() -> np.ndarray:
            rng = np.random.default_rng(abs(hash(subject_id + segment_id)) % (2**32))
            eeg = rng.normal(0.0, 1.0, size=(self.config.eeg_num_channels, target_time)).astype(np.float32)
            return eeg

        # Find candidate .mat files matching subject id digits
        digits_only = ''.join(ch for ch in subject_id if ch.isdigit())
        candidates: List[str] = []
        for root, _dirs, files in os.walk(EEG_DATASET_DIR):
            for f in files:
                if f.lower().endswith('.mat') and digits_only and digits_only in f:
                    candidates.append(os.path.join(root, f))
        # Weak fallback: if nothing matched, just grab any .mat to proceed
        if not candidates:
            for root, _dirs, files in os.walk(EEG_DATASET_DIR):
                for f in files:
                    if f.lower().endswith('.mat'):
                        candidates.append(os.path.join(root, f))
                        break
                if candidates:
                    break
        if not candidates:
            return _fallback()

        mat_path = candidates[0]
        try:
            mat = sio.loadmat(mat_path)
        except Exception:
            return _fallback()

        # Try common keys
        possible_keys = [
            'data', 'EEG', 'eeg', 'signal', 'signals', 'X', 'x', 'Data'
        ]
        arr: Optional[np.ndarray] = None
        for k in possible_keys:
            if k in mat:
                val = mat[k]
                if isinstance(val, np.ndarray) and val.size > 0:
                    arr = val
                    break
        if arr is None:
            return _fallback()

        # Ensure 2D: (channels, time) or (time, channels)
        eeg_arr = np.array(arr)
        eeg_arr = np.squeeze(eeg_arr)
        if eeg_arr.ndim == 1:
            eeg_arr = eeg_arr[None, :]
        elif eeg_arr.ndim > 2:
            # Try to pick a 2D slice if 3D
            eeg_arr = eeg_arr.reshape(eeg_arr.shape[0], -1)

        # Heuristic: if rows >> cols, assume (time, channels)
        if eeg_arr.shape[0] > eeg_arr.shape[1]:
            eeg_arr = eeg_arr.T

        # Now eeg_arr is (channels, time)
        # Select first N channels
        eeg_arr = eeg_arr[: self.config.eeg_num_channels]

        # Center-crop/pad to target_time
        if eeg_arr.shape[1] >= target_time:
            start = (eeg_arr.shape[1] - target_time) // 2
            eeg_arr = eeg_arr[:, start:start + target_time]
        else:
            pad = target_time - eeg_arr.shape[1]
            eeg_arr = np.pad(eeg_arr, ((0, 0), (0, pad)), mode='constant')

        return eeg_arr.astype(np.float32)

    def _load_label(self, subject_id: str) -> int:
        """Placeholder for loading subject label.

        Replace with actual MODMA labels mapping. 1=Depression, 0=Healthy.
        """
        # Simple deterministic pseudo-label for placeholder
        return 1 if (abs(hash(subject_id)) % 2 == 0) else 0

    # ----------------------------
    # Preprocessing placeholder stubs
    # ----------------------------
    def _preprocess_speech(self, audio_data: np.ndarray, subject_id: str) -> np.ndarray:
        """Convert raw audio into features.

        Options: MFCCs, log-Mel spectrograms, wav2vec2 embeddings.
        Current placeholder: log-Mel-like random features with per-speaker normalization.

        Returns shape (feat_dim, time_steps).
        """
        # Placeholder: generate pseudo log-Mel features
        feat_dim = 80
        time_steps = max(8, len(audio_data) // (self.config.sample_rate // 10))
        rng = np.random.default_rng(len(audio_data))
        features = rng.normal(0.0, 1.0, size=(feat_dim, time_steps)).astype(np.float32)

        if self.config.speech_normalize_per_speaker:
            mean_per_feature = features.mean(axis=1, keepdims=True)
            std_per_feature = features.std(axis=1, keepdims=True) + 1e-6
            features = (features - mean_per_feature) / std_per_feature

        return features

    def _preprocess_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """Select channels, band-pass filter, window, and normalize per channel.

        Placeholder implements shape handling and normalization only.
        Returns shape (num_channels=64, time_steps).
        """
        # Select first N channels
        num_channels = min(self.config.eeg_num_channels, eeg_data.shape[0])
        eeg = eeg_data[:num_channels]

        # Placeholder band-pass: No-op. Replace with real DSP (e.g., scipy.signal)

        # Placeholder windowing: return contiguous segment. Replace with sliding windows if needed.

        if self.config.eeg_normalize_per_channel:
            mean_per_channel = eeg.mean(axis=1, keepdims=True)
            std_per_channel = eeg.std(axis=1, keepdims=True) + 1e-6
            eeg = (eeg - mean_per_channel) / std_per_channel

        return eeg


# ==================
# Model Architectures
# ==================


class SpeechEncoder(nn.Module):
    """CNN + BiLSTM speech encoder.

    Input: (batch, feat_dim, time)
    Output: (batch, hidden) pooled embedding and (batch, time, lstm_hidden*2) features
    """

    def __init__(self, input_feat_dim: int = 80, cnn_channels: int = 64, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_feat_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, F, T)
        x = self.conv(x)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)
        lstm_out = self.dropout(lstm_out)
        # Temporal pooling to get utterance-level embedding
        pooled = torch.mean(lstm_out, dim=1)  # (B, 2*H)
        return pooled, lstm_out


class EEGEncoder(nn.Module):
    """1D CNN + BiLSTM EEG encoder.

    Input: (batch, channels=64, time)
    Output: (batch, hidden) pooled embedding and (batch, time, lstm_hidden*2) features
    """

    def __init__(self, input_channels: int = 64, cnn_channels: int = 64, lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        x = self.conv(x)  # (B, C', T)
        x = x.transpose(1, 2)  # (B, T, C')
        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)
        lstm_out = self.dropout(lstm_out)
        pooled = torch.mean(lstm_out, dim=1)  # (B, 2*H)
        return pooled, lstm_out


class SelfAttentionBlock(nn.Module):
    """Simple self-attention over sequence features.

    Expects input shape (B, T, D). Returns (B, D).
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))  # (B, T, T)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)  # (B, T, D)
        pooled = attended.mean(dim=1)  # (B, D)
        return pooled


class ImprovedFeatureFusionBlock(nn.Module):
    """Improved Feature Fusion Block (IFFB) with attention.

    Combines speech and EEG embeddings and optionally leverages sequence features.

    Inputs:
        speech_seq: (B, T_s, D_s)
        eeg_seq:    (B, T_e, D_e)
        speech_emb: (B, D_s)
        eeg_emb:    (B, D_e)

    Output:
        fused: (B, D_fused)
    """

    def __init__(self, speech_dim: int, eeg_dim: int, hidden_dim: int = 256, fused_dim: int = 256, use_sequence_attention: bool = True) -> None:
        super().__init__()
        self.use_sequence_attention = use_sequence_attention
        self.s_attn = SelfAttentionBlock(speech_dim)
        self.e_attn = SelfAttentionBlock(eeg_dim)
        self.fc = nn.Sequential(
            nn.Linear(speech_dim + eeg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, fused_dim),
            nn.ReLU(),
        )

    def forward(self, speech_seq: torch.Tensor, eeg_seq: torch.Tensor, speech_emb: torch.Tensor, eeg_emb: torch.Tensor) -> torch.Tensor:
        if self.use_sequence_attention:
            s_ctx = self.s_attn(speech_seq)  # (B, D_s)
            e_ctx = self.e_attn(eeg_seq)     # (B, D_e)
            concat = torch.cat([speech_emb + s_ctx, eeg_emb + e_ctx], dim=-1)
        else:
            concat = torch.cat([speech_emb, eeg_emb], dim=-1)
        fused = self.fc(concat)
        return fused


class TeacherModel(nn.Module):
    """Teacher model: Speech + EEG with IFFB and classifier.

    Forward returns:
        logits: (B, 1)
        extras: dict with intermediate features for KD
            - speech_features: (B, T_s, D_s)
            - speech_embedding: (B, D_s)
            - eeg_features: (B, T_e, D_e)
            - eeg_embedding: (B, D_e)
            - fused: (B, D_fused)
    """

    def __init__(self, speech_feat_dim: int = 80, eeg_channels: int = 64, lstm_hidden: int = 128) -> None:
        super().__init__()
        self.speech_encoder = SpeechEncoder(input_feat_dim=speech_feat_dim, lstm_hidden=lstm_hidden)
        self.eeg_encoder = EEGEncoder(input_channels=eeg_channels, lstm_hidden=lstm_hidden)
        speech_out_dim = lstm_hidden * 2
        eeg_out_dim = lstm_hidden * 2
        self.iffb = ImprovedFeatureFusionBlock(speech_dim=speech_out_dim, eeg_dim=eeg_out_dim, hidden_dim=256, fused_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, speech_x: torch.Tensor, eeg_x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # speech_x: (B, F, T)
        # eeg_x:    (B, C, T)
        speech_emb, speech_seq = self.speech_encoder(speech_x)
        eeg_emb, eeg_seq = self.eeg_encoder(eeg_x)
        fused = self.iffb(speech_seq, eeg_seq, speech_emb, eeg_emb)
        logits = self.classifier(fused)
        extras = {
            "speech_features": speech_seq,
            "speech_embedding": speech_emb,
            "eeg_features": eeg_seq,
            "eeg_embedding": eeg_emb,
            "fused": fused,
        }
        return logits, extras


class StudentModel(nn.Module):
    """Student model: Speech-only encoder + classifier.

    Forward returns:
        logits: (B, 1)
        extras: dict with intermediate features for KD
            - speech_features: (B, T_s, D_s)
            - speech_embedding: (B, D_s)
    """

    def __init__(self, speech_feat_dim: int = 80, lstm_hidden: int = 128) -> None:
        super().__init__()
        self.speech_encoder = SpeechEncoder(input_feat_dim=speech_feat_dim, lstm_hidden=lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, speech_x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        speech_emb, speech_seq = self.speech_encoder(speech_x)
        logits = self.classifier(speech_emb)
        extras = {
            "speech_features": speech_seq,
            "speech_embedding": speech_emb,
        }
        return logits, extras


# ================================
# Knowledge Distillation Loss (KD)
# ================================


def kd_total_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    ground_truth_labels: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total KD loss.

    Components:
        - Classification: BCEWithLogitsLoss(student_logits, labels)
        - Logit distillation: KLDivLoss(log_softmax(student), softmax(teacher))
        - Feature distillation: MSELoss(student_features, teacher_features)

    Returns total loss and dict of components.
    """
    bce = nn.BCEWithLogitsLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    mse = nn.MSELoss()

    # Ensure correct shapes for BCE: (B, 1) vs (B,) labels
    labels = ground_truth_labels.view_as(student_logits)

    classification_loss = bce(student_logits, labels)

    # Convert logits to 2-class probabilities for KL on binary. Stack logits for [p0, p1].
    # Alternatively, treat as single logit; here we form 2-class by [1-logit, logit].
    student_prob = torch.cat([torch.sigmoid(-student_logits), torch.sigmoid(student_logits)], dim=-1)
    teacher_prob = torch.cat([torch.sigmoid(-teacher_logits), torch.sigmoid(teacher_logits)], dim=-1)

    # Use log_softmax/softmax flavor over the constructed 2-class dim
    log_student = torch.log(student_prob + 1e-8)
    soft_teacher = teacher_prob
    logit_distill_loss = kl(log_student, soft_teacher)

    # Align feature dimensions if needed with a simple projection (optional upgrade)
    if student_features.shape != teacher_features.shape:
        # Basic temporal alignment: interpolate student to teacher time or vice versa
        # Here, we project time dimension by interpolation if 3D tensors
        if student_features.dim() == 3 and teacher_features.dim() == 3:
            # (B, T, D)
            if student_features.size(1) != teacher_features.size(1):
                student_features_interp = F.interpolate(
                    student_features.transpose(1, 2), size=teacher_features.size(1), mode="linear", align_corners=False
                ).transpose(1, 2)
            else:
                student_features_interp = student_features
            # Match feature dim by a linear projection-on-the-fly
            if student_features_interp.size(2) != teacher_features.size(2):
                # Create projection dynamically; not learnable in loss, but avoids crash
                # Stop-grad on weights by using detach-based least-squares-like matching is overkill; use pad/crop.
                sf, tf = student_features_interp, teacher_features
                if sf.size(2) > tf.size(2):
                    student_features_matched = sf[..., : tf.size(2)]
                else:
                    pad = tf.size(2) - sf.size(2)
                    student_features_matched = F.pad(sf, (0, pad))
            else:
                student_features_matched = student_features_interp
        else:
            student_features_matched = student_features
    else:
        student_features_matched = student_features

    feature_distill_loss = mse(student_features_matched, teacher_features.detach())

    total = alpha * classification_loss + beta * logit_distill_loss + gamma * feature_distill_loss
    components = {
        "classification": float(classification_loss.detach().cpu().item()),
        "logit_distillation": float(logit_distill_loss.detach().cpu().item()),
        "feature_distillation": float(feature_distill_loss.detach().cpu().item()),
    }
    return total, components


# ===========================
# Training Procedure Functions
# ===========================


def _compute_batch_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    probs = torch.sigmoid(logits.detach()).cpu().numpy().reshape(-1)
    preds = (probs >= 0.5).astype(np.int32)
    y_true = labels.detach().cpu().numpy().reshape(-1)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    return acc, f1


def train_teacher_model(
    model: TeacherModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_f1 = 0.0
        num_batches = 0
        for speech_x, eeg_x, labels in train_dataloader:
            speech_x = speech_x.to(device)
            eeg_x = eeg_x.to(device)
            labels = labels.to(device).view(-1, 1)

            optimizer.zero_grad()
            logits, _ = model(speech_x, eeg_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc, f1 = _compute_batch_metrics(logits, labels)
            total_loss += float(loss.detach().cpu().item())
            total_acc += acc
            total_f1 += f1
            num_batches += 1

        train_loss = total_loss / max(1, num_batches)
        train_acc = total_acc / max(1, num_batches)
        train_f1 = total_f1 / max(1, num_batches)

        if val_dataloader is not None:
            val_metrics = evaluate_model(model, val_dataloader, device)
            print(f"[Teacher][Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}")
        else:
            print(f"[Teacher][Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")


def train_student_model_with_kd(
    student_model: StudentModel,
    teacher_model: TeacherModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    kd_loss_fn,
    alpha: float,
    beta: float,
    gamma: float,
    device: torch.device,
) -> None:
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad = False

    for epoch in range(1, epochs + 1):
        student_model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_f1 = 0.0
        num_batches = 0
        for speech_x, eeg_x, labels in train_dataloader:
            speech_x = speech_x.to(device)
            eeg_x = eeg_x.to(device)
            labels = labels.to(device).view(-1, 1)

            optimizer.zero_grad()

            # Teacher forward (no grads)
            with torch.no_grad():
                t_logits, t_extras = teacher_model(speech_x, eeg_x)
                t_features = t_extras["speech_features"]

            # Student forward
            s_logits, s_extras = student_model(speech_x)
            s_features = s_extras["speech_features"]

            # KD loss
            loss, components = kd_loss_fn(
                student_logits=s_logits,
                teacher_logits=t_logits,
                student_features=s_features,
                teacher_features=t_features,
                ground_truth_labels=labels,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            loss.backward()
            optimizer.step()

            acc, f1 = _compute_batch_metrics(s_logits, labels)
            total_loss += float(loss.detach().cpu().item())
            total_acc += acc
            total_f1 += f1
            num_batches += 1

        train_loss = total_loss / max(1, num_batches)
        train_acc = total_acc / max(1, num_batches)
        train_f1 = total_f1 / max(1, num_batches)

        if val_dataloader is not None:
            val_metrics = evaluate_model(student_model, val_dataloader, device, is_student=True)
            print(
                f"[Student-KD][Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
            )
        else:
            print(f"[Student-KD][Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")


# =====================
# Evaluation
# =====================


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, is_student: bool = False) -> Dict[str, float]:
    model.eval()
    model.to(device)

    all_probs: List[float] = []
    all_labels: List[int] = []

    for speech_x, eeg_x, labels in dataloader:
        speech_x = speech_x.to(device)
        labels = labels.to(device).view(-1, 1)
        if is_student:
            logits, _ = model(speech_x)
        else:
            eeg_x = eeg_x.to(device)
            logits, _ = model(speech_x, eeg_x)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().reshape(-1).tolist())

    preds = (np.array(all_probs) >= 0.5).astype(np.int32)
    y_true = np.array(all_labels).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, np.array(all_probs)) if len(np.unique(y_true)) > 1 else 0.5),
    }
    return metrics


# =====================
# Optional: CE Student
# =====================


def train_student_ce(
    student_model: StudentModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    student_model.to(device)
    for epoch in range(1, epochs + 1):
        student_model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_f1 = 0.0
        num_batches = 0
        for speech_x, _eeg_x, labels in train_dataloader:
            speech_x = speech_x.to(device)
            labels = labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            logits, _ = student_model(speech_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            acc, f1 = _compute_batch_metrics(logits, labels)
            total_loss += float(loss.detach().cpu().item())
            total_acc += acc
            total_f1 += f1
            num_batches += 1

        train_loss = total_loss / max(1, num_batches)
        train_acc = total_acc / max(1, num_batches)
        train_f1 = total_f1 / max(1, num_batches)

        if val_dataloader is not None:
            val_metrics = evaluate_model(student_model, val_dataloader, device, is_student=True)
            print(f"[Student-CE][Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}")
        else:
            print(f"[Student-CE][Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f}")


# =====================
# Main Script Structure
# =====================


def _build_dataloaders(config: DatasetConfig, batch_size: int = 8, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    train_ds = DepressionDataset(config, split="train")
    val_ds = DepressionDataset(config, split="val")

    # Collate: pad/time-align if necessary. Here tensors are same-sized placeholders.
    def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        speech_list, eeg_list, label_list = zip(*batch)
        speech_batch = torch.stack(speech_list, dim=0)  # (B, F, T)
        eeg_batch = torch.stack(eeg_list, dim=0)        # (B, C, T)
        label_batch = torch.stack(label_list, dim=0)    # (B,)
        return speech_batch, eeg_batch, label_batch

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    return train_loader, val_loader


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the speech dataset directory as the root for subject discovery.
    # In a full implementation, you'd align entries across SPEECH_DATASET_DIR and EEG_DATASET_DIR by subject IDs.
    config = DatasetConfig(
        root_dir=SPEECH_DATASET_DIR,
        speech_dirname="speech",  # placeholder; adapt if you restructure into per-subject subfolders
        eeg_dirname="eeg",        # placeholder; EEG alignment logic would map to EEG_DATASET_DIR
        sample_rate=16000,
        eeg_num_channels=64,
        speech_feature="logmel",
        window_seconds=5.0,
        window_stride_seconds=1.0,
    )

    # Optional smoke test path: set SMOKE_TEST=1 to only run one forward pass
    if os.environ.get("SMOKE_TEST") == "1":
        print("Running smoke test (single batch forward pass)...")
        train_loader, val_loader = _build_dataloaders(config, batch_size=2)
        # Get one batch
        it = iter(train_loader)
        try:
            speech_x, eeg_x, labels = next(it)
        except StopIteration:
            print("No data found for smoke test.")
            raise SystemExit(0)
        teacher = TeacherModel(speech_feat_dim=80, eeg_channels=config.eeg_num_channels, lstm_hidden=64)
        student = StudentModel(speech_feat_dim=80, lstm_hidden=64)
        teacher.to(device)
        student.to(device)
        speech_x = speech_x.to(device)
        eeg_x = eeg_x.to(device)
        with torch.no_grad():
            t_logits, t_extras = teacher(speech_x, eeg_x)
            s_logits, s_extras = student(speech_x)
        print("Teacher logits:", tuple(t_logits.shape))
        print("Student logits:", tuple(s_logits.shape))
        print("Speech feature seq:", tuple(t_extras["speech_features"].shape))
        print("EEG feature seq:", tuple(t_extras["eeg_features"].shape))
        raise SystemExit(0)

    # Data
    train_loader, val_loader = _build_dataloaders(config, batch_size=8)

    # Models
    teacher = TeacherModel(speech_feat_dim=80, eeg_channels=config.eeg_num_channels, lstm_hidden=128)
    student = StudentModel(speech_feat_dim=80, lstm_hidden=128)

    # Optimizers and criteria
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    # Train Teacher
    print("Training Teacher...")
    train_teacher_model(teacher, train_loader, val_loader, epochs=1, optimizer=teacher_optimizer, criterion=bce, device=device)

    # Train Student with KD
    print("Training Student with KD...")
    train_student_model_with_kd(
        student_model=student,
        teacher_model=teacher,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
        optimizer=student_optimizer,
        kd_loss_fn=kd_total_loss,
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
        device=device,
    )

    # Evaluate
    print("Evaluating Teacher...")
    teacher_metrics = evaluate_model(teacher, val_loader, device)
    print("Teacher:", teacher_metrics)

    print("Evaluating Student (KD)...")
    student_metrics = evaluate_model(student, val_loader, device, is_student=True)
    print("Student-KD:", student_metrics)

    # Optional: Train Student without KD (comparison)
    # print("Training Student with CE only...")
    # student_ce = StudentModel(speech_feat_dim=80, lstm_hidden=128)
    # student_ce_opt = torch.optim.Adam(student_ce.parameters(), lr=1e-3)
    # train_student_ce(student_ce, train_loader, val_loader, epochs=1, optimizer=student_ce_opt, criterion=bce, device=device)
    # print("Evaluating Student (CE)...")
    # student_ce_metrics = evaluate_model(student_ce, val_loader, device, is_student=True)
    # print("Student-CE:", student_ce_metrics)


