# Blood Pressure Estimation from PPG Signals

## Project Goal
Build a Machine Learning pipeline to estimate **Systolic Blood Pressure (SBP)** and **Diastolic Blood Pressure (DBP)** using Photoplethysmogram (PPG) signals extracted from video.

---

## Dataset: UCI Cuff-less Blood Pressure Estimation

**Location:** `./uci_dataset/`

### Files
| File | Size | Description |
|------|------|-------------|
| `Part_1.mat` | ~851 MB | Patient records batch 1 |
| `Part_2.mat` | ~973 MB | Patient records batch 2 |
| `Part_3.mat` | ~718 MB | Patient records batch 3 |
| `Part_4.mat` | ~867 MB | Patient records batch 4 |

### Data Structure
Each `.mat` file contains a cell array (typically named `p`). Each cell represents a single patient record containing a **3×N matrix** of synchronized signals:

| Row Index | Signal | Role | Sampling Rate |
|-----------|--------|------|---------------|
| 0 | **PPG** (Photoplethysmogram) | Model Input (X) | 125 Hz |
| 1 | **ABP** (Arterial Blood Pressure) | Target Source (y) | 125 Hz |
| 2 | **ECG** (Electrocardiogram) | Not used | 125 Hz |

### Target Label Extraction
The ABP signal is a continuous waveform (in mmHg). To create discrete BP labels:
- **Systolic BP (SBP):** Peak values in the ABP waveform (heart contraction)
- **Diastolic BP (DBP):** Valley/trough values in the ABP waveform (heart relaxation)

---

## Signal Processing Concepts

### PPG (Photoplethysmogram)
An optical measurement of blood volume changes in tissue. When captured via camera:
- The **green channel** provides the best signal (hemoglobin absorption)
- Typical frequency content: 0.5–8 Hz (filters out motion artifacts and high-frequency noise)

### Key PPG Morphological Features
| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| Pulse Width | Duration of the pulse wave | Arterial stiffness indicator |
| Upstroke Time | Rise time from valley to peak | Vascular compliance |
| Peak-to-Peak Interval | Time between consecutive peaks | Heart rate variability |
| Dicrotic Notch | Secondary peak from aortic valve closure | Vascular health marker |
| Pulse Area | Area under the PPG waveform | Blood volume estimation |

### Realistic BP Ranges (for data validation)
| Measurement | Normal | Elevated | Hypertensive |
|-------------|--------|----------|--------------|
| Systolic | 90–120 | 120–139 | ≥140 |
| Diastolic | 60–80 | 80–89 | ≥90 |

**Valid data range for filtering:** 60–180 mmHg (discard outliers outside this range)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│  .mat files → Data Loader → 5s Windows → Feature Extraction → Model    │
│       ↓              ↓           ↓              ↓              ↓        │
│   Part_*.mat    UCILoader    PPG + ABP    NeuroKit2 Features   RF Reg   │
│                              synchronized                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Video → Green Channel → Resample to 125Hz → Features → Model → BP     │
│    ↓          ↓               ↓                 ↓         ↓      ↓      │
│  .mp4     OpenCV        scipy.resample      NeuroKit2   Trained  SBP   │
│                                                         Model    DBP   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Steps

1. **Data Ingestion** (`data_loader.py`)
   - Load MATLAB files using `scipy.io.loadmat()`
   - Slice continuous signals into 5-second non-overlapping windows
   - Extract SBP/DBP targets from ABP peaks/valleys per window

2. **Feature Engineering** (`features.py`)
   - Bandpass filter: 0.5–8 Hz (Butterworth)
   - Extract morphological features using NeuroKit2
   - Handle noisy segments gracefully (return None for invalid windows)

3. **Model Training** (`model_trainer.py`)
   - Algorithm: `RandomForestRegressor`
   - Multi-output regression: predicts both SBP and DBP
   - Evaluation metric: Mean Absolute Error (MAE)

4. **Inference** (`inference_video.py`)
   - Extract PPG from video green channel
   - Resample from camera FPS (30/60 Hz) to 125 Hz
   - Apply identical feature extraction pipeline

---

## Technical Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12+ |
| ML | Scikit-learn (RandomForestRegressor) |
| Signal Processing | NeuroKit2, SciPy |
| Data Handling | NumPy, Pandas |
| Video Processing | OpenCV (cv2) |
| MATLAB Files | scipy.io |
| Progress | tqdm |

---

## Key Considerations

### Critical Implementation Notes
1. **Sampling rate consistency:** Training data is 125 Hz. Video data must be resampled to match.
2. **Feature alignment:** Save feature column names with model—inference must use identical features.
3. **Window size:** 5 seconds × 125 Hz = 625 samples per window.
4. **Noise handling:** Discard windows where NeuroKit2 fails to detect peaks.

### Validation Criteria for Windows
- BP values within realistic range (60–180 mmHg)
- Minimum number of peaks detected in window
- Signal quality check (not flat, not excessive noise)

---

## References
- Project Documentation: `README.md` (contains calibration guide and CNN implementation details)
- UCI Dataset: [Cuff-less Blood Pressure Estimation](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
- NeuroKit2 Documentation: [PPG Analysis](https://neuropsychology.github.io/NeuroKit/functions/ppg.html)