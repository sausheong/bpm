# Blood Pressure Estimation from PPG Signals

A machine learning pipeline to estimate **Systolic (SBP)** and **Diastolic (DBP)** blood pressure from Photoplethysmogram (PPG) signals extracted from smartphone videos.

This project implements two approaches:
1.  **Feature Engineering + Random Forest:** Robust on small data, interpretable.
2.  **1D CNN (Deep Learning):** End-to-end learning from raw signals, optimized for RTX GPUs (FP16/AMP).

## 🎯 Quick Start

### 1. Prerequisites

- **Python 3.12+**
- **NVIDIA GPU** (Recommended for CNN)
- **uv** (Recommended for fast dependency management)

### 2. Install Dependencies

Using `uv`:
```bash
uv pip install -r requirements.txt
```

> **Note for RTX 5090 Users:** This project uses PyTorch 2.9.1+ and NumPy 1.26.4 to ensure compatibility with Blackwell architecture and CUDA 12.8+.

### 3. Train Models

#### Option A: Random Forest (Fast, CPU-friendly)
Best for quick prototyping or small datasets.
```bash
uv run python train.py --model rf --parts 1 2 3 4
```

#### Option B: 1D CNN (High Performance, GPU-accelerated)
Best for large datasets and raw signal learning.
```bash
uv run python train.py --model cnn --parts 1 2 3 4 --epochs 100 --batch-size 32
```
*Supports Automatic Mixed Precision (AMP) for reduced memory usage.*

### 4. Inference (Predict from Video)

Predict BP using your smartphone camera:
```bash
# Uses Random Forest by default
uv run python predict_bp.py your_video.mp4

# Use CNN model
uv run python predict_bp.py your_video.mp4 --model cnn
```

**Video Requirements:**
- Place finger covering the **entire** main camera and flash.
- Record for **20-30 seconds**.
- Keep finger steady and relaxed.

---

## ⚖️ Model Comparison

| Feature | Random Forest (RF) | 1D CNN |
|---------|-------------------|--------|
| **Input** | Hand-crafted features (HRV, peaks) | Raw PPG waveform (625 samples) |
| **Training Time** | Fast (< 5 mins) | Slow (~1 hour on GPU) |
| **Hardware** | CPU | NVIDIA GPU (Recommended) |
| **Data Efficiency** | High (Works well with scanty data) | Low (Needs large datasets) |
| **Pros** | Interpretable, stable | Learn complex non-linear patterns |
| **Cons** | Feature engineering bottle-neck | Compute intensive |

---

## 🛠️ Calibration

For personalized accuracy, you can calibrate the model with a reference measurement.

```bash
# 1. Measure BP with a cuff (e.g., 120/80)
# 2. Record video immediately
# 3. Run calibration
uv run python predict_bp.py calibration_video.mp4 --calibrate --sbp 120 --dbp 80
```
See [CALIBRATION.md](CALIBRATION.md) for details.

---

## 🏗️ Project Structure

```
bpm/
├── src/
│   ├── data_loader.py      # UCI dataset loading (HDF5)
│   ├── features.py         # NeuroKit2 feature extraction
│   ├── model_cnn.py        # PyTorch 1D CNN Architecture
│   ├── model_trainer.py    # RF Training Pipeline
│   ├── model_trainer_cnn.py# CNN Training Pipeline
│   └── inference_video.py  # Video processing & prediction
├── model/
│   ├── bp_model.pkl        # Saved RF Model
│   └── bp_model_cnn.pt     # Saved CNN Weights
├── uci_dataset/            # Place Part_1.mat, etc. here
├── train.py                # Main training entry point
├── predict_bp.py           # Inference entry point
└── requirements.txt
```

## 📝 Technical Details

- **Signal Processing**: All signals resampled to **125 Hz**.
- **Windowing**: Non-overlapping **5-second windows**.
- **Filtering**: Butterworth Bandpass (0.5 - 8 Hz).
- **CNN Architecture**: 
  - 4x Conv1D blocks with BatchNorm & ReLU.
  - Global Average Pooling.
  - 2x Dense layers with Dropout (0.3).
  - Trained with L1 Loss (MAE) and Adam Optimizer.

## 📄 License
Research use only. Cite UCI Cuff-less Blood Pressure Estimation Dataset.
