# Blood Pressure Estimation from PPG Signals

This project provides a machine learning pipeline to estimate Systolic (SBP) and Diastolic (DBP) blood pressure from Photoplethysmogram (PPG) signals extracted from smartphone videos. It implements both a feature-engineered Random Forest (RF) baseline and a deep learning 1D Convolutional Neural Network (CNN).

## Quick Start

### Prerequisites
- Python 3.12+
- NVIDIA GPU (Recommended for CNN training and inference)
- uv (Recommended for dependency management)

### Installation
Clone the repository and install dependencies using `uv`:
```bash
uv pip install -r requirements.txt
```

### Training
Train the models on the UCI dataset parts. The script auto-detects available hardware (CUDA/MPS/CPU).

```bash
# Train Random Forest (CPU-based)
uv run python train.py --model rf --parts 1 2 3 4

# Train 1D CNN (Hardware-accelerated)
uv run python train.py --model cnn --parts 1 2 3 4 --epochs 100
```

### Prediction
Predict blood pressure from a recorded video:
```bash
# Default prediction using Random Forest
uv run python predict.py your_video.mp4

# Prediction using CNN (Recommended for accuracy)
uv run python predict.py your_video.mp4 --model cnn
```

## Model Performance

The models were evaluated on the UCI Cuff-less Blood Pressure Estimation dataset.

| Metric | Random Forest | 1D CNN |
|--------|--------------|--------|
| SBP MAE | 13.91 mmHg | 7.50 mmHg |
| DBP MAE | 6.67 mmHg | 4.18 mmHg |
| SBP R2 | 0.24 | 0.70 |
| DBP R2 | 0.56 | 0.74 |
| Hardware | CPU | GPU (CUDA/MPS) |

## Calibration System

The system includes a calibration mechanism to personalize predictions using reference measurements from clinical devices. This accounts for individual physiological variations and model biases.

### Calibration Commands
```bash
# Calibrate model with a reference BP reading
uv run python predict.py video.mp4 --calibrate --sbp 120 --dbp 80 --model cnn

# View current calibration status
uv run python predict.py --show-calibration

# Clear all stored calibration data
uv run python predict.py --clear-calibration
```

## Technical Specifications

### 1D CNN Architecture
The CNN model processes raw PPG signals through a series of convolutional layers designed for temporal feature extraction:
- Input: (1, 625) normalized PPG signal
- Layers:
    - Conv1D (32 filters, kernel=7) + BatchNorm + ReLU + MaxPool
    - Conv1D (64 filters, kernel=5) + BatchNorm + ReLU + MaxPool
    - Conv1D (128 filters, kernel=3) + BatchNorm + ReLU + MaxPool
    - Conv1D (128 filters, kernel=3) + BatchNorm + ReLU
    - GlobalAveragePooling1D
    - Fully Connected (128) + ReLU + Dropout (0.3)
    - Fully Connected (64) + ReLU + Dropout (0.3)
    - Output: Fully Connected (2) -> [SBP, DBP]
- Parameters: ~110K trainable weights

### Signal Processing Pipeline
- Windowing: 5-second segments at 125 Hz (625 samples per window)
- Filtering: Butterworth Bandpass filter (0.5–8.0 Hz)
- Normalization: StandardScaler applied per signal (CNN) or per feature (RF)
- Video Processing: Automatic color channel selection based on signal-to-noise ratio; resampling from camera FPS to 125 Hz.

### Cross-Platform Support
The project utilizes PyTorch for the CNN implementation and automatically detects the best available backend:
- CUDA: NVIDIA GPUs (via `cuda`)
- MPS: Apple Silicon GPUs (via `mps`)
- CPU: Standard fallback for all systems

## Project Structure

- `src/`: Core implementation modules
    - `model_cnn.py`: CNN architecture and inference logic
    - `model_trainer.py`: Random Forest training orchestration
    - `model_trainer_cnn.py`: CNN training orchestration
    - `data_loader.py`: UCI dataset parsing and windowing
    - `features.py`: PPG feature extraction using NeuroKit2
    - `calibration.py`: Persistent calibration management
    - `inference_video.py`: End-to-end video-to-BP pipeline
- `model/`: Directory for saved model weights and metadata
- `uci_dataset/`: Target directory for .mat data files
- `train.py`: CLI for model training
- `predict.py`: CLI for running inference on video files

## References
- UCI Cuff-less Blood Pressure Estimation Dataset: [Archive Link](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
- Signal analysis facilitated by [NeuroKit2](https://neuropsychology.github.io/NeuroKit/)


