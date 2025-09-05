# Teacher Model for Speech Emotion Recognition (SER)

## Knowledge Distillation Project - Teacher Model Implementation

This repository contains the Teacher Model implementation for a Speech Emotion Recognition system using Knowledge Distillation. The Teacher Model is designed to be a larger, more accurate model that will later transfer its knowledge to a smaller Student Model.

## 🎯 Project Overview

- **Purpose**: Build a robust Teacher Model for Speech Emotion Recognition
- **Architecture**: CNN + BiLSTM with comprehensive feature extraction
- **Datasets**: Compatible with RAVDESS, CREMA-D, SAVEE datasets
- **Framework**: PyTorch with MPS (Metal Performance Shaders) support for MacBook Air M2

## 🏗️ Model Architecture

### Teacher Model Components:
1. **Feature Extraction Layer**: 
   - MFCCs (40 coefficients)
   - Chroma features (12 coefficients)
   - Mel-spectrogram (128 bands)
   - Zero Crossing Rate
   - Spectral Centroid, Rolloff, and Bandwidth

2. **CNN Layers**:
   - Conv1D + BatchNorm + ReLU + MaxPooling
   - Conv1D + BatchNorm + ReLU + MaxPooling

3. **BiLSTM Layer**:
   - Bidirectional LSTM with configurable hidden size and layers
   - Dropout for regularization

4. **Dense Layers**:
   - Fully connected layers with ReLU activation
   - Softmax output for emotion classification

## 📋 Requirements

### System Requirements:
- macOS (optimized for MacBook Air M2)
- Python 3.8+
- Sufficient RAM for audio processing

### Dependencies:
Install all required packages using:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Dataset Preparation

Place your audio dataset in a directory with the following structure:
```
your_dataset/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-01-02.wav
│   └── ...
├── Actor_02/
│   └── ...
└── ...
```

### 2. Configuration

Update the `DATA_DIR` variable in `teacher_model.py`:
```python
DATA_DIR = "path/to/your/audio/dataset"
```

### 3. Run Training

Execute the Teacher Model training:
```bash
python teacher_model.py
```

## 📊 Features

### ✅ Implemented Features:

1. **Audio Feature Extraction**:
   - MFCCs, Chroma, Mel-spectrogram
   - Zero Crossing Rate, Spectral features
   - Automatic feature vector creation

2. **Data Preprocessing**:
   - Feature normalization using StandardScaler
   - Label encoding with sklearn LabelEncoder
   - Train/Validation/Test split (80/10/10)

3. **Model Training**:
   - Adam optimizer with learning rate scheduling
   - CrossEntropyLoss for multi-class classification
   - Early stopping based on validation accuracy
   - Model checkpointing (saves best model)

4. **Evaluation & Visualization**:
   - Classification report (precision, recall, F1-score)
   - Confusion matrix visualization
   - Training/validation curves plotting
   - Accuracy metrics

5. **Model Persistence**:
   - Saves trained model weights (`teacher_model.pth`)
   - Saves preprocessing objects (`preprocessing_objects.pkl`)

## 📈 Expected Outputs

### Files Generated:
- `teacher_model.pth`: Best trained model weights
- `preprocessing_objects.pkl`: Scaler and label encoder for inference
- `confusion_matrix.png`: Confusion matrix visualization
- `training_curves.png`: Training and validation curves

### Console Output:
- Training progress with epoch-wise metrics
- Final test accuracy
- Classification report
- Model parameter count

## 🔧 Customization

### Model Hyperparameters:
```python
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
```

### Feature Extraction Parameters:
```python
n_mfcc = 40      # Number of MFCC coefficients
n_chroma = 12    # Number of chroma features
n_mels = 128     # Number of mel bands
sr = 22050       # Sample rate
```

## 🎯 Emotion Classes

Default emotion mapping (RAVDESS format):
- `01`: neutral
- `02`: calm
- `03`: happy
- `04`: sad
- `05`: angry
- `06`: fearful
- `07`: disgust
- `08`: surprised

## 🔄 Next Steps

After training the Teacher Model:

1. **Student Model Development**: Create a smaller, more efficient model
2. **Knowledge Distillation**: Transfer knowledge from Teacher to Student
3. **Streamlit Deployment**: Deploy the Student Model in a web application
4. **Performance Comparison**: Compare Teacher vs Student model performance

## 🐛 Troubleshooting

### Common Issues:

1. **MPS Device Not Available**:
   - The code automatically falls back to CPU if MPS is not available
   - Ensure you have PyTorch 2.0+ installed

2. **Audio File Loading Errors**:
   - Check file formats (supports .wav, .mp3, .flac)
   - Ensure audio files are not corrupted

3. **Memory Issues**:
   - Reduce batch size in DataLoader
   - Process dataset in smaller chunks

4. **Dataset Path Issues**:
   - Verify the DATA_DIR path is correct
   - Ensure audio files are in the expected directory structure

## 📝 Code Structure

```
teacher_model.py
├── AudioDataset (PyTorch Dataset)
├── TeacherModel (CNN + BiLSTM architecture)
├── extract_features() (Audio feature extraction)
├── load_dataset() (Dataset loading and processing)
├── preprocess_data() (Data preprocessing and splitting)
├── create_model() (Model instantiation)
├── train_model() (Training loop)
├── evaluate_model() (Model evaluation)
├── plot_training_curves() (Visualization)
└── main() (Complete pipeline)
```

## 🤝 Contributing

This is a final year college project. Feel free to:
- Report bugs and issues
- Suggest improvements
- Contribute to the knowledge distillation pipeline

## 📄 License

This project is created for educational purposes as part of a final year college project.

---

**Note**: Make sure to update the `DATA_DIR` path before running the script. The model is optimized for MacBook Air M2 but will work on other systems with appropriate device configuration.
