# Blood Pressure Estimation from PPG Signals

A machine learning pipeline to estimate **Systolic (SBP)** and **Diastolic (DBP)** blood pressure from Photoplethysmogram (PPG) signals extracted from smartphone videos.

## 🎯 Quick Start

### 1. Install Dependencies

Using `uv` (recommended):
```bash
uv pip install -r requirements.txt
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 2. Train the Model

Debug mode (fast, uses ~100 windows):
```bash
uv run python train.py --max-windows 1000
```

Full training (all data from Part 1):
```bash
uv run python train.py --full
```

### 3. Predict Blood Pressure from Video

```bash
uv run python predict_bp.py your_video.mp4
```

**Video Requirements:**
- Place finger on phone camera for 20-30 seconds
- Keep finger still and relaxed
- Ensure good lighting
- MP4 format

### 4. Calibrate for Personalized Results (Optional but Recommended)

For improved accuracy, calibrate with a reference BP measurement:

```bash
# Measure BP with a validated device (e.g., automated cuff)
# Then record a video immediately and run:
uv run python predict_bp.py calibration_video.mp4 --calibrate --sbp 120 --dbp 80

# Future predictions will automatically use calibration
uv run python predict_bp.py new_video.mp4
# Output shows: "✓ Calibrated prediction"
```

**See [CALIBRATION.md](CALIBRATION.md) for detailed calibration guide**

---

## 📊 Performance

Trained on UCI Cuff-less Blood Pressure Estimation Dataset:

| Metric | Training | Test |
|--------|----------|------|
| **SBP MAE** | 2.05 mmHg | 4.74 mmHg ✓ |
| **DBP MAE** | 1.26 mmHg | 2.71 mmHg ✓ |
| **Overall MAE** | 1.65 mmHg | 3.73 mmHg |

Target: SBP < 15 mmHg, DBP < 12 mmHg ✅

---

## 🏗️ Project Structure

```
bpm/
├── src/
│   ├── data_loader.py      # UCI dataset loading (HDF5 support)
│   ├── features.py         # PPG feature extraction (NeuroKit2)
│   ├── model_trainer.py    # Training pipeline (RandomForest)
│   └── inference_video.py  # Video-based BP prediction
├── model/
│   └── bp_model.pkl        # Trained model + metadata
├── uci_dataset/
│   └── Part_*.mat          # Training data (not included)
├── train.py           # Training entry point
├── predict_bp.py           # Inference entry point
├── requirements.txt
├── context.md              # Project documentation
└── plan.md                 # Implementation guide
```

---

## 🔬 How It Works

### Training Pipeline

```
.mat files → Data Loader → 5s Windows → Feature Extraction → RandomForest
```

1. **Data Loading** (`data_loader.py`)
   - Loads MATLAB v7.3 (HDF5) files
   - Extracts PPG and ABP signals
   - Slices into 5-second windows (625 samples @ 125Hz)
   - Detects peaks/valleys in ABP for SBP/DBP labels

2. **Feature Extraction** (`features.py`)
   - Normalizes PPG signal for peak detection
   - Cleans signal (bandpass filter 0.5-8 Hz)
   - Extracts 11 morphological features:
     - Heart rate (mean, std)
     - HRV time-domain (MeanNN, SDNN, RMSSD)
     - Pulse amplitude (mean, std)
     - Signal statistics

3. **Model Training** (`model_trainer.py`)
   - RandomForest regressor (100 trees)
   - Multi-output regression (predicts both SBP and DBP)
   - Saves model with feature names and scaler

### Inference Pipeline

```
Video → Best Channel → Resample to 125Hz → Features → Model → BP
```

1. **PPG Extraction** (`inference_video.py`)
   - Reads video frame-by-frame
   - Extracts mean intensity from RGB channels
   - **Automatically selects channel with highest variance**
   - (Green preferred, but red/blue used if better)

2. **Resampling**
   - Converts video FPS (typically 30) to 125 Hz
   - Uses scipy.signal.resample for anti-aliasing

3. **Feature Extraction**
   - Processes multiple 5-second windows
   - Averages features for robustness

4. **Prediction**
   - Loads trained model
   - Aligns features with training columns
   - Returns SBP/DBP with confidence score

---

## 💡 Example Output

```
Processing video: bpm.mp4
[1/4] Extracting PPG signal from green channel...
  Selected Red channel (variance: 52.27)
  Extracted 944 samples at 30.0 FPS
  Duration: 31.5 seconds
[2/4] Resampling to 125 Hz...
  Resampled signal: 3937 samples
[3/4] Extracting features...
  Extracted 11 features
[4/4] Loading model and predicting...

==================================================
BLOOD PRESSURE ESTIMATION RESULT
==================================================
Systolic BP (SBP):  127.8 mmHg
Diastolic BP (DBP): 69.0 mmHg
Signal Quality:     Fair (confidence: 0.60)
==================================================
```

---

## 🎛️ Advanced Usage

### Train on Multiple Dataset Parts

```bash
uv run python train.py --parts 1 2 --full
```

### Custom Output Path

```bash
uv run python train.py --output models/my_model.pkl
```

### Direct Python API

```python
from src.inference_video import video_to_bp

result = video_to_bp("video.mp4", "model/bp_model.pkl")
print(f"BP: {result['sbp']}/{result['dbp']} mmHg")
print(f"Quality: {result['signal_quality']}")
```

---

## 📝 Technical Details

### Signal Processing
- **Sampling Rate:** 125 Hz (all signals normalized to this)
- **Window Size:** 5 seconds (625 samples)
- **Filter:** Bandpass 0.5-8 Hz (removes baseline wander and high-freq noise)
- **Peak Detection:** NeuroKit2 derivative-based method with fallback to Elgendi

### Features Extracted
| Feature | Description |
|---------|-------------|
| PPG_Rate_Mean | Average heart rate (BPM) |
| PPG_Rate_Std | Heart rate variability |
| HRV_MeanNN | Mean time between peaks (ms) |
| HRV_SDNN | Standard deviation of NN intervals |
| HRV_RMSSD | Root mean square of successive differences |
| PPG_Peaks_Amplitude_Mean | Average peak amplitude |
| PPG_Peaks_Amplitude_Std | Peak amplitude variability |
| PPG_Mean/Std/Range | Basic signal statistics |
| PPG_NumPeaks | Number of detected peaks |

### Model Configuration
- **Algorithm:** RandomForestRegressor (scikit-learn)
- **Trees:** 100
- **Max Depth:** 15
- **Min Samples Split:** 5
- **Features:** Standardized (StandardScaler)

---

## ⚠️ Important Notes

1. **Not Medical Advice:** This is a research/educational tool, not a medical device
2. **Video Quality:** Signal quality depends heavily on:
   - Finger placement (fully covering camera)
   - Stability (hold still)
   - Lighting conditions
   - Skin tone and blood perfusion
3. **Calibration:** Model trained on UCI dataset; results may vary for different populations
4. **Channel Selection:** Automatically uses red channel if green is unavailable/poor

---

## 📚 Dataset

UCI Cuff-less Blood Pressure Estimation Dataset:
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
- **Format:** MATLAB v7.3 (HDF5)
- **Signals:** PPG, ABP, ECG
- **Sampling Rate:** 125 Hz
- **Patients:** 3000 records

Place dataset files in `uci_dataset/` directory:
- `Part_1.mat` (~851 MB)
- `Part_2.mat` (~973 MB)
- `Part_3.mat` (~718 MB)
- `Part_4.mat` (~867 MB)

---

## 🔧 Dependencies

- `numpy` >= 1.24
- `pandas` >= 2.0
- `scipy` >= 1.11
- `scikit-learn` >= 1.3
- `neurokit2` >= 0.2
- `opencv-python` >= 4.8
- `h5py` >= 3.0 (for MATLAB v7.3 files)
- `tqdm` >= 4.65
- `joblib` >= 1.3
- `matplotlib` >= 3.7

---

## 📖 References

- NeuroKit2: [PPG Analysis Documentation](https://neuropsychology.github.io/NeuroKit/functions/ppg.html)
- Photoplethysmography: Non-invasive optical measurement of blood volume changes
- Heart Rate Variability: Time-domain and frequency-domain analysis

---

## 🤝 Contributing

This is a research project. For questions or improvements, please refer to `context.md` and `plan.md` for detailed technical specifications.

---

## 📄 License

Educational/Research use. Please cite the UCI dataset if used in publications.
