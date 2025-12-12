# Blood Pressure Estimation from PPG Signals

A machine learning pipeline to estimate **Systolic (SBP)** and **Diastolic (DBP)** blood pressure from Photoplethysmogram (PPG) signals extracted from smartphone videos.

## Quick Start

### Prerequisites
- **Python 3.12+**
- **NVIDIA GPU** (Recommended for CNN)
- **uv** (Recommended for dependency management)

### Install
```bash
uv pip install -r requirements.txt
```

### Train
```bash
# Random Forest (fast, CPU)
uv run python train.py --model rf --parts 1 2 3 4

# 1D CNN (GPU-accelerated)
uv run python train.py --model cnn --parts 1 2 3 4 --epochs 100
```

### Predict
```bash
uv run python predict_bp.py your_video.mp4
```

---

## Model Comparison

| Feature | Random Forest | 1D CNN |
|---------|--------------|--------|
| **SBP MAE** | 13.91 mmHg | **7.50 mmHg** (Best) |
| **DBP MAE** | 6.67 mmHg | **4.18 mmHg** (Best) |
| **R² Score** | 0.24 | **0.70** |
| **Hardware** | CPU | GPU (CUDA) |

---

## Calibration

Personalize predictions with a reference measurement:

```bash
# Calibrate with reference BP reading
uv run python predict_bp.py video.mp4 --calibrate --sbp 120 --dbp 80

# View/clear calibration
uv run python predict_bp.py --show-calibration
uv run python predict_bp.py --clear-calibration
```

---

## API Reference

### CNN Model (src/model_cnn.py)

```python
from src.model_cnn import create_cnn_model, train_cnn_model, predict_with_cnn

# Create model
model = create_cnn_model()

# Train
model, history = train_cnn_model(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32
)

# Predict
predictions = predict_with_cnn(model, ppg_signals)  # (N, 2) [SBP, DBP]
```

### Training Pipelines

```python
# Random Forest
from src.model_trainer import train_pipeline, compute_metrics
metrics = train_pipeline(
    data_path=['uci_dataset/Part_1.mat'],
    output_path='model/bp_model.pkl'
)

# CNN
from src.model_trainer_cnn import train_cnn_pipeline
metrics = train_cnn_pipeline(
    data_path=['uci_dataset/Part_1.mat'],
    epochs=100, batch_size=32
)
```

---

## Technical Details

### CNN Architecture
```
Input (1, 625)
→ Conv1D(32, k=7) + BN + ReLU + MaxPool
→ Conv1D(64, k=5) + BN + ReLU + MaxPool
→ Conv1D(128, k=3) + BN + ReLU + MaxPool
→ Conv1D(128, k=3) + BN + ReLU
→ GlobalAvgPool → FC(128) → FC(64) → FC(2)
```
**Parameters:** ~110K trainable

### Signal Processing
- **Window**: 5 seconds @ 125 Hz (625 samples)
- **Normalization**: StandardScaler (mean=0, std=1)
- **Filtering**: Butterworth Bandpass (0.5–8 Hz)

### Training
- **Loss**: L1 (MAE)
- **Optimizer**: Adam + ReduceLROnPlateau
- **Regularization**: Dropout (0.3), BatchNorm
- **GPU**: FP16 AMP on CUDA

---

## Project Structure

```
bpm/
├── src/
│   ├── model_cnn.py          # CNN architecture & training
│   ├── model_trainer.py      # RF training pipeline
│   ├── model_trainer_cnn.py  # CNN training pipeline
│   ├── data_loader.py        # UCI dataset loader
│   ├── features.py           # NeuroKit2 feature extraction
│   ├── calibration.py        # User calibration
│   └── inference_video.py    # Video → BP prediction
├── model/                    # Saved models
├── uci_dataset/              # Training data (.mat files)
├── train.py                  # CLI training entry
├── predict_bp.py             # CLI inference entry
└── requirements.txt
```

---

## References

- [UCI Cuff-less BP Dataset](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
- [NeuroKit2 PPG Analysis](https://neuropsychology.github.io/NeuroKit/functions/ppg.html)

## License
Research use only. Cite UCI dataset if used in publications.
