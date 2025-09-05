import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import pickle
from typing import Tuple, List, Dict, Optional, Any

warnings.filterwarnings('ignore')

# Set device for MacBook Air M2
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class AudioDataset(Dataset):
    """Custom Dataset for loading audio features and labels"""
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class TeacherModel(nn.Module):
    """Teacher Model Architecture: CNN + BiLSTM for Speech Emotion Recognition"""
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3) -> None:
        super(TeacherModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 4  # After 2 max pooling layers
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dense layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 256)  # *2 for bidirectional
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input for Conv1d: (batch, channels, features)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # Reshape for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from LSTM
        x = lstm_out[:, -1, :]
        
        # Dense layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def extract_features(audio_path: str, sr: int = 22050, n_mfcc: int = 40, 
                   n_chroma: int = 12, n_mels: int = 128) -> Optional[Dict[str, np.ndarray]]:
    """Extract audio features from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        features: Dict[str, np.ndarray] = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features['mfccs'] = np.mean(mfccs, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        features['chroma'] = np.mean(chroma, axis=1)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        features['mel_spec'] = np.mean(mel_spec, axis=1)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr'] = np.mean(zcr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid'] = np.mean(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def load_dataset(data_dir: str, emotion_mapping: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load audio dataset and extract features"""
    features_list: List[np.ndarray] = []
    labels_list: List[str] = []
    
    # Default emotion mapping for RAVDESS dataset
    if emotion_mapping is None:
        emotion_mapping = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
    
    print("Loading dataset and extracting features...")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(root, file)
                
                try:
                    # Extract emotion from filename (RAVDESS format)
                    if '-' in file:
                        parts = file.split('-')
                        if len(parts) >= 3:
                            emotion_code = parts[2]
                            emotion = emotion_mapping.get(emotion_code, 'unknown')
                        else:
                            continue
                    else:
                        emotion = 'unknown'
                        for code, emo in emotion_mapping.items():
                            if code in file:
                                emotion = emo
                                break
                    
                    features = extract_features(file_path)
                    if features is not None:
                        feature_vector = np.concatenate([
                            features['mfccs'],
                            features['chroma'],
                            features['mel_spec'],
                            [features['zcr']],
                            [features['spectral_centroid']],
                            [features['spectral_rolloff']],
                            [features['spectral_bandwidth']]
                        ])
                        
                        features_list.append(feature_vector)
                        labels_list.append(emotion)
                        
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
    
    print(f"Loaded {len(features_list)} samples with {len(set(labels_list))} emotion classes")
    return np.array(features_list), np.array(labels_list)

def load_lanzhou_audio_dataset(
    data_dir: str,
    metadata_path: str,
    label_column: str = 'type',
    max_subjects: Optional[int] = None,
    max_files_per_subject: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Lanzhou audio dataset using metadata labels (e.g., MDD/HC).

    Expects directory structure:
        data_dir/
            <subject_folder_numeric>/
                01.wav, 02.wav, ...

    And an Excel metadata file with at least columns: 'subject id', label_column.

    Subject folder names like '02010002' are mapped to metadata 'subject id' by
    converting to int then back to string (drops leading zeros): '02010002' -> '2010002'.
    """

    print("Loading Lanzhou dataset and extracting features (labels from metadata)...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata Excel not found: {metadata_path}")

    # Read metadata and normalize subject ids to string
    meta_df = pd.read_excel(metadata_path)
    if 'subject id' not in meta_df.columns:
        raise ValueError("Expected column 'subject id' in metadata Excel")
    if label_column not in meta_df.columns:
        raise ValueError(f"Expected label column '{label_column}' in metadata Excel")

    # Clean columns (strip, unify types)
    meta_df['subject id'] = meta_df['subject id'].astype(str).str.strip()
    meta_df[label_column] = meta_df[label_column].astype(str).str.strip()

    # Build mapping from subject id -> label
    subject_to_label: Dict[str, str] = (
        meta_df.set_index('subject id')[label_column].to_dict()  # type: ignore[assignment]
    )

    # Enumerate subject folders
    subject_folders = [
        d for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d)) and any(ch.isdigit() for ch in d)
    ]

    features_list: List[np.ndarray] = []
    labels_list: List[str] = []

    processed_subjects = 0
    for subject_folder in subject_folders:
        # Map folder name to metadata subject id by dropping leading zeros via int casting
        normalized_subject_id = None
        try:
            digits_only = ''.join(ch for ch in subject_folder if ch.isdigit())
            if len(digits_only) == 0:
                continue
            normalized_subject_id = str(int(digits_only))
        except Exception:
            # Skip folders that cannot be normalized
            continue

        # Lookup label
        label = subject_to_label.get(normalized_subject_id)
        if label is None:
            # Not present in metadata, skip
            # print(f"Warning: Subject {subject_folder} (id {normalized_subject_id}) not in metadata; skipping")
            continue

        subject_path = os.path.join(data_dir, subject_folder)
        audio_files = [
            f for f in sorted(os.listdir(subject_path))
            if f.lower().endswith(('.wav', '.flac', '.mp3'))
        ]

        if max_files_per_subject is not None:
            audio_files = audio_files[: max_files_per_subject]

        for file in audio_files:
            file_path = os.path.join(subject_path, file)
            try:
                features = extract_features(file_path)
                if features is None:
                    continue

                feature_vector = np.concatenate([
                    features['mfccs'],
                    features['chroma'],
                    features['mel_spec'],
                    [features['zcr']],
                    [features['spectral_centroid']],
                    [features['spectral_rolloff']],
                    [features['spectral_bandwidth']],
                ])

                features_list.append(feature_vector)
                labels_list.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        processed_subjects += 1
        if max_subjects is not None and processed_subjects >= max_subjects:
            break

    print(
        f"Loaded {len(features_list)} samples from {processed_subjects} subjects "
        f"with labels: {sorted(set(labels_list))}"
    )

    return np.array(features_list), np.array(labels_list)

def preprocess_data(features: np.ndarray, labels: np.ndarray, test_size: float = 0.2, 
                   val_size: float = 0.125) -> Tuple[DataLoader, DataLoader, DataLoader, 
                                                    StandardScaler, LabelEncoder, int, int]:
    """Preprocess data: normalize, encode labels, and split into train/val/test sets"""
    print("Preprocessing data...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Split into train/test first
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_normalized, labels_encoded, test_size=test_size, 
        random_state=42, stratify=labels_encoded
    )
    
    # Split remaining data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, 
        random_state=42, stratify=y_temp
    )
    
    # Create datasets and loaders
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_size = features.shape[1]
    num_classes = len(label_encoder.classes_)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    return train_loader, val_loader, test_loader, scaler, label_encoder, input_size, num_classes

def create_model(input_size: int, num_classes: int, hidden_size: int = 128, 
                num_layers: int = 2, dropout: float = 0.3) -> TeacherModel:
    """Create Teacher Model"""
    model = TeacherModel(input_size, num_classes, hidden_size, num_layers, dropout)
    return model.to(device)

def train_model(model: TeacherModel, train_loader: DataLoader, val_loader: DataLoader, 
               num_epochs: int = 30, learning_rate: float = 0.001) -> Tuple[List[float], List[float], 
                                                                           List[float], List[float]]:
    """Train the Teacher Model"""
    print("Training Teacher Model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses: List[float] = []
    train_accuracies: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'teacher_model.pth')
            print(f"Saved best model with validation accuracy: {val_accuracy:.2f}%")
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model: TeacherModel, test_loader: DataLoader, 
                  label_encoder: LabelEncoder) -> Tuple[float, List[str], List[str]]:
    """Evaluate the trained Teacher Model"""
    print("Evaluating Teacher Model...")
    
    model.eval()
    all_predictions: List[int] = []
    all_labels: List[int] = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    y_true = label_encoder.inverse_transform(all_labels)
    y_pred = label_encoder.inverse_transform(all_predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Teacher Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, y_true, y_pred

def plot_training_curves(train_losses: List[float], train_accuracies: List[float], 
                        val_losses: List[float], val_accuracies: List[float]) -> None:
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main() -> None:
    """Main function to run the complete Teacher Model pipeline"""
    print("=== TEACHER MODEL FOR SPEECH EMOTION RECOGNITION ===")
    print("Knowledge Distillation Project - Teacher Model Implementation")
    print("=" * 60)
    
    # Configuration
    # Set to your Lanzhou audio dataset directory and metadata Excel path
    DATA_DIR = "/Users/atharvpareta/Desktop/major/audio_lanzhou_2015"
    METADATA_XLSX = "/Users/atharvpareta/Desktop/major/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx"
    LABEL_COLUMN = 'type'  # 'type' contains MDD/HC labels in the metadata

    # Optional: quick smoke-test limits (set to None for full run)
    MAX_SUBJECTS = None  # e.g., 3
    MAX_FILES_PER_SUBJECT = None  # e.g., 3
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        print("Please update the DATA_DIR variable with the correct path to your audio dataset.")
        print("Expected format: Lanzhou 'audio_lanzhou_2015' subject folders with wav files")
        return
    if not os.path.exists(METADATA_XLSX):
        print(f"Error: Metadata Excel '{METADATA_XLSX}' not found!")
        print("Please update the METADATA_XLSX variable with the correct path to the Lanzhou metadata file.")
        return
    
    try:
        # Step 1: Load dataset and extract features
        print("\n1. Loading dataset and extracting features (Lanzhou)...")
        features, labels = load_lanzhou_audio_dataset(
            DATA_DIR,
            METADATA_XLSX,
            label_column=LABEL_COLUMN,
            max_subjects=MAX_SUBJECTS,
            max_files_per_subject=MAX_FILES_PER_SUBJECT,
        )
        
        if len(features) == 0:
            print("No valid audio files found in the dataset directory!")
            return
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        train_loader, val_loader, test_loader, scaler, label_encoder, input_size, num_classes = preprocess_data(features, labels)
        
        # Step 3: Create model
        print("\n3. Creating Teacher Model...")
        model = create_model(input_size, num_classes, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Step 4: Train model
        print("\n4. Training Teacher Model...")
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE
        )
        
        # Step 5: Plot training curves
        print("\n5. Plotting training curves...")
        plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
        
        # Step 6: Load best model and evaluate
        print("\n6. Loading best model and evaluating...")
        model.load_state_dict(torch.load('teacher_model.pth'))
        accuracy, y_true, y_pred = evaluate_model(model, test_loader, label_encoder)
        
        # Step 7: Save preprocessing objects for later use
        print("\n7. Saving preprocessing objects...")
        with open('preprocessing_objects.pkl', 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'label_encoder': label_encoder,
                'input_size': input_size,
                'num_classes': num_classes
            }, f)
        
        print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"Best model saved as: teacher_model.pth")
        print(f"Preprocessing objects saved as: preprocessing_objects.pkl")
        print(f"Final test accuracy: {accuracy:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
