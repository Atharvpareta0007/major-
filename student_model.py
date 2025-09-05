import os
import warnings
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse data pipeline and utilities from the teacher module
from teacher_model import (
    load_lanzhou_audio_dataset,
    preprocess_data,
    evaluate_model,
    create_model as create_teacher_model,
)


warnings.filterwarnings('ignore')

# Device configuration (optimized for MacBook Air M2 with MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device (student): {device}")


class StudentModel(nn.Module):
    """Lightweight MLP Student Model for speech-only features.

    Designed to be distilled from the TeacherModel via knowledge distillation.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        hidden1, hidden2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_student_model(
    input_size: int,
    num_classes: int,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.2,
) -> StudentModel:
    model = StudentModel(input_size, num_classes, hidden_dims, dropout)
    return model.to(device)


class DistillationLoss(nn.Module):
    """Knowledge Distillation loss combining CE (hard) and KL (soft)."""

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature
        # Hard loss
        ce_loss = self.ce(student_logits, true_labels)
        # Soft loss: KL(student||teacher) with softened distributions
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
        # Weighted sum
        return self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute standard KD loss = alpha * T^2 * KL + (1 - alpha) * CE.

    Returns: (total_loss, ce_loss, kl_loss)
    """
    # Cross-entropy with hard labels
    ce = F.cross_entropy(student_logits, true_labels)

    # KL divergence between softened probabilities
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)

    total = alpha * kl + (1.0 - alpha) * ce
    return total, ce.detach(), kl.detach()


def train_student_with_kd(
    student: StudentModel,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 0.001,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train student with KD from the teacher.

    Returns: train_losses, train_accuracies, val_losses, val_accuracies
    """
    print("Training Student Model with Knowledge Distillation...")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    # Use integrated DistillationLoss
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)

    train_losses: List[float] = []
    train_accuracies: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        student.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(features)
            student_logits = student(features)

            total_loss = criterion(student_logits, teacher_logits, labels)

            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        train_accuracy = 100.0 * train_correct / max(1, train_total)

        # Validation phase
        student.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                teacher_logits = teacher(features)
                student_logits = student(features)
                total_loss = criterion(student_logits, teacher_logits, labels)

                epoch_val_loss += total_loss.item()
                _, predicted = torch.max(student_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = epoch_val_loss / max(1, len(val_loader))
        val_accuracy = 100.0 * val_correct / max(1, val_total)

        scheduler.step(avg_val_loss)

        # Track
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Save best
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(student.state_dict(), 'student_model.pth')
            print(f"Saved best student model with validation accuracy: {val_accuracy:.2f}%")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

    print(f"Student training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return train_losses, train_accuracies, val_losses, val_accuracies


def main() -> None:
    print("=== STUDENT MODEL (Speech-only) WITH KNOWLEDGE DISTILLATION ===")
    print("Reference: TeacherModel (speech) -> StudentModel (speech)")
    print("=" * 60)

    # Configuration (adjust as needed)
    DATA_DIR = "/Users/atharvpareta/Desktop/major/audio_lanzhou_2015"
    METADATA_XLSX = "/Users/atharvpareta/Desktop/major/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx"

    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    TEMPERATURE = 4.0
    ALPHA = 0.7  # weight for distillation (1-alpha for CE)

    # Optional: subset for quick iteration
    MAX_SUBJECTS: Optional[int] = None  # e.g., 3
    MAX_FILES_PER_SUBJECT: Optional[int] = None  # e.g., 3

    # Paths
    TEACHER_WEIGHTS = 'teacher_model.pth'

    # Checks
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        return
    if not os.path.exists(METADATA_XLSX):
        print(f"Error: Metadata Excel '{METADATA_XLSX}' not found!")
        return

    # Load dataset
    print("\n1. Loading dataset and extracting features (Lanzhou)...")
    features, labels = load_lanzhou_audio_dataset(
        DATA_DIR,
        METADATA_XLSX,
        label_column='type',
        max_subjects=MAX_SUBJECTS,
        max_files_per_subject=MAX_FILES_PER_SUBJECT,
    )
    if len(features) == 0:
        print("No samples found. Please check your dataset path.")
        return

    # Preprocess
    print("\n2. Preprocessing data...")
    train_loader, val_loader, test_loader, scaler, label_encoder, input_size, num_classes = preprocess_data(
        features, labels
    )

    # Create teacher
    print("\n3. Loading Teacher Model...")
    teacher = create_teacher_model(input_size, num_classes)
    if os.path.exists(TEACHER_WEIGHTS):
        try:
            teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=device))
            print("Teacher weights loaded.")
        except Exception as e:
            print(f"Warning: Failed to load teacher weights: {e}")
            print("Proceeding without distillation (CE only).")
            ALPHA = 0.0
    else:
        print("Warning: teacher_model.pth not found. Proceeding with CE-only training.")
        ALPHA = 0.0

    # Create student
    print("\n4. Creating Student Model...")
    student = create_student_model(input_size, num_classes, hidden_dims=(128, 64), dropout=0.2)
    print(f"Student model params: {sum(p.numel() for p in student.parameters()):,}")

    # Train student (with KD if ALPHA>0)
    print("\n5. Training Student Model...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_student_with_kd(
        student,
        teacher,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
        alpha=ALPHA,
    )

    # Evaluate
    print("\n6. Evaluating Student Model (best checkpoint)...")
    try:
        student.load_state_dict(torch.load('student_model.pth', map_location=device))
    except Exception:
        print("Note: Best checkpoint not found; evaluating current student weights.")
    accuracy, y_true, y_pred = evaluate_model(student, test_loader, label_encoder)

    print("\n=== STUDENT TRAINING COMPLETED ===")
    print(f"Best student model saved as: student_model.pth")
    print(f"Final student test accuracy: {accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


