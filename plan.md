# Implementation Plan: BP Estimation Pipeline

> **Role:** Senior ML Engineer  
> **Objective:** Scaffold a robust Python project for blood pressure estimation from PPG signals

---

## Project Structure

```
bpm/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # UCI dataset loading and windowing
│   ├── features.py         # PPG feature extraction
│   ├── model_trainer.py    # Training pipeline
│   └── inference_video.py  # Video-based BP prediction
├── model/
│   └── bp_model.pkl        # Trained model + feature names
├── uci_dataset/
│   └── Part_*.mat          # Training data
├── train.py           # Training entry point
├── requirements.txt
└── README.md
```

---

## File Specifications

### 1. `src/data_loader.py`

**Class:** `UCILoader`

```python
class UCILoader:
    """Load UCI Cuff-less BP dataset and generate windowed samples."""
    
    SAMPLING_RATE = 125  # Hz
    WINDOW_SECONDS = 5
    WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS  # 625 samples
    BP_MIN, BP_MAX = 60, 180  # Valid BP range in mmHg
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `load_part(filename)` | Path to `.mat` file | `(X_raw, y)` | Load and process one file |
| `_window_signals(ppg, abp)` | Raw signals | List of (ppg_window, sbp, dbp) | Slice into 5s windows |
| `_extract_bp_targets(abp_window)` | ABP window | (mean_sbp, mean_dbp) or None | Detect peaks/valleys |

**Critical Logic:**

```python
def _extract_bp_targets(self, abp_window):
    """
    Extract SBP and DBP from ABP waveform.
    
    ABP Signal Anatomy:
    - Peaks = Systolic pressure (max arterial pressure during contraction)
    - Valleys = Diastolic pressure (min arterial pressure during relaxation)
    
    Algorithm:
    1. Detect peaks using scipy.signal.find_peaks (prominence-based)
    2. Detect valleys by finding peaks in inverted signal
    3. Calculate mean SBP from peaks, mean DBP from valleys
    4. Validate: discard if outside [60, 180] mmHg range
    """
    # Implementation here
```

**Quality Checks:**
- [x] Discard windows with < 3 peaks detected
- [x] Discard windows with BP outside 60–180 mmHg
- [x] Skip flat or noisy signals (std < threshold)

---

### 2. `src/features.py`

**Class:** `PPGFeatureExtractor`

```python
class PPGFeatureExtractor:
    """Extract morphological features from PPG signals using NeuroKit2."""
    
    def __init__(self, sampling_rate: int = 125):
        self.sampling_rate = sampling_rate
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `process_window(ppg_signal)` | 1D numpy array | dict or None | Extract features from single window |
| `transform(X_raw_list)` | List of windows | pd.DataFrame | Batch process with multiprocessing |

**Feature Extraction Steps:**

```python
def process_window(self, ppg_signal: np.ndarray) -> Optional[dict]:
    """
    Process a single PPG window.
    
    Steps:
    1. Clean signal: nk.ppg_clean(signal, sampling_rate)
       - Applies bandpass filter (0.5-8 Hz by default)
       - Removes baseline wander and high-frequency noise
    
    2. Find peaks: nk.ppg_findpeaks(cleaned, sampling_rate)
       - Uses derivative-based peak detection
       - Returns indices of systolic peaks
    
    3. Analyze: nk.ppg_analyze(peaks, sampling_rate)
       - Extracts: PPG_Rate_Mean, HRV metrics, morphology
       
    Error Handling:
    - Return None if < 3 peaks detected (insufficient for analysis)
    - Return None if NeuroKit raises exception (noisy signal)
    """
```

**Expected Features (from NeuroKit2):**

| Category | Features |
|----------|----------|
| Rate | PPG_Rate_Mean |
| HRV Time | HRV_RMSSD, HRV_SDNN, HRV_MeanNN |
| HRV Frequency | HRV_LF, HRV_HF, HRV_LFHF |
| Morphology | Peak amplitudes, intervals |

**Multiprocessing Wrapper:**

```python
def transform(self, X_raw_list: List[np.ndarray], n_workers: int = None) -> pd.DataFrame:
    """
    Process multiple windows in parallel.
    
    Args:
        X_raw_list: List of raw PPG windows (each 625 samples)
        n_workers: Number of parallel workers (default: CPU count - 1)
    
    Returns:
        DataFrame with feature columns. Rows with failed extraction are dropped.
        Also returns valid_indices to align with y labels.
    """
```

---

### 3. `src/model_trainer.py`

**Training Pipeline:**

```python
def train_pipeline(data_path: str, output_path: str = "model/bp_model.pkl"):
    """
    Complete training pipeline.
    
    Steps:
    1. Load data using UCILoader
    2. Extract features using PPGFeatureExtractor
    3. Align X and y (drop samples where feature extraction failed)
    4. Train/test split (80/20)
    5. Train RandomForestRegressor
    6. Evaluate and print MAE
    7. Save model + feature names
    """
```

**Model Configuration:**

```python
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)
```

**Saved Artifacts:**

```python
# Save both model and feature column names
joblib.dump({
    'model': trained_model,
    'feature_names': X_train.columns.tolist(),
    'scaler': scaler  # Optional: if standardizing features
}, output_path)
```

**Evaluation Metrics:**

```python
def evaluate(y_true, y_pred):
    """Print MAE for SBP and DBP separately."""
    mae_sbp = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_dbp = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    print(f"SBP MAE: {mae_sbp:.2f} mmHg")
    print(f"DBP MAE: {mae_dbp:.2f} mmHg")
```

---

### 4. `src/inference_video.py`

**Function:** `video_to_bp(video_path: str) -> Tuple[float, float]`

```python
def video_to_bp(video_path: str, model_path: str = "model/bp_model.pkl") -> dict:
    """
    Predict blood pressure from smartphone video.
    
    Args:
        video_path: Path to .mp4 file (finger placed on camera)
        model_path: Path to trained model pickle
    
    Returns:
        dict with 'sbp', 'dbp', 'confidence' keys
    
    Pipeline:
    1. Extract PPG signal from video
    2. Resample to 125 Hz
    3. Extract features (identical to training)
    4. Predict using loaded model
    """
```

**Video Processing:**

```python
def extract_ppg_from_video(video_path: str) -> Tuple[np.ndarray, float]:
    """
    Extract PPG signal from video green channel.
    
    Why Green Channel?
    - Hemoglobin has peak absorption at ~540nm (green light)
    - Blood volume changes cause measurable intensity variations
    
    Algorithm:
    1. Read video frame by frame using cv2.VideoCapture
    2. For each frame: extract mean intensity of green channel
    3. Return time series + original FPS
    
    Returns:
        (signal, fps): 1D numpy array of green intensities, video framerate
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    green_values = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        green_channel = frame[:, :, 1]  # BGR format
        green_values.append(np.mean(green_channel))
    
    cap.release()
    return np.array(green_values), fps
```

**Resampling:**

```python
def resample_to_target(signal: np.ndarray, original_fps: float, target_fs: int = 125):
    """
    Resample signal to match training data frequency.
    
    Example:
    - Video at 30 FPS, 10 seconds → 300 samples
    - Target: 125 Hz, 10 seconds → 1250 samples
    
    Uses scipy.signal.resample for anti-aliased resampling.
    """
    duration = len(signal) / original_fps
    target_samples = int(duration * target_fs)
    return scipy.signal.resample(signal, target_samples)
```

---

### 5. `train.py`

**Training Entry Point:**

```python
#!/usr/bin/env python3
"""
Blood Pressure Estimation - Training Script

Usage:
    python train.py                    # Debug mode (1000 windows)
    python train.py --full             # Full training
    python train.py --parts 1 2        # Train on specific parts
"""

import argparse
from src.data_loader import UCILoader
from src.features import PPGFeatureExtractor
from src.model_trainer import train_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Use full dataset')
    parser.add_argument('--parts', nargs='+', type=int, default=[1], 
                        help='Which Part_*.mat files to use')
    parser.add_argument('--max-windows', type=int, default=1000,
                        help='Max windows for debugging (ignored if --full)')
    args = parser.parse_args()
    
    # ... training logic
```

---

## Implementation Constraints

### Required Libraries
```
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
neurokit2>=0.2
opencv-python>=4.8
tqdm>=4.65
joblib>=1.3
```

### Code Quality Requirements

- [ ] Use `tqdm` for all loops over data
- [ ] Add comprehensive docstrings explaining signal processing math
- [ ] Type hints on all function signatures
- [ ] Handle edge cases gracefully (noisy signals, empty windows)
- [ ] Log warnings for skipped samples (don't fail silently)

### Critical Invariants

1. **Sampling Rate Consistency**
   ```python
   # Training data: 125 Hz
   # Video data: varies (30/60 FPS)
   # MUST resample video data to 125 Hz before feature extraction
   ```

2. **Feature Column Alignment**
   ```python
   # During inference, features must match training columns exactly
   # Load saved feature_names and reindex DataFrame
   X_inference = X_inference.reindex(columns=saved_feature_names, fill_value=0)
   ```

3. **Window Synchronization**
   ```python
   # PPG and ABP windows must be perfectly aligned
   # Start index for both signals must be identical
   ppg_window = ppg[i:i+WINDOW_SIZE]
   abp_window = abp[i:i+WINDOW_SIZE]  # Same indices!
   ```

---

## Execution Order

1. **Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Debug Run**
   ```bash
   python train.py --max-windows 1000
   ```

3. **Full Training**
   ```bash
   python train.py --full --parts 1 2 3 4
   ```

4. **Inference Test**
   ```bash
   python -c "from src.inference_video import video_to_bp; print(video_to_bp('bpm.mp4'))"
   ```

---

## Expected Results

| Metric | Target | Acceptable |
|--------|--------|------------|
| SBP MAE | < 10 mmHg | < 15 mmHg |
| DBP MAE | < 8 mmHg | < 12 mmHg |
| Training Time (1000 windows) | < 5 min | < 10 min |