# Blood Pressure Calibration Guide

## Overview

The calibration feature allows you to personalize BP predictions using reference measurements from a clinically validated blood pressure device (e.g., automated cuff monitor).

## Why Calibrate?

- **Improved Accuracy**: Accounts for individual physiological differences
- **Personalized Offsets**: Learns your specific BP patterns
- **Better Estimates**: Reduces systematic errors in predictions

---

## How to Calibrate

### Step 1: Measure with Reference Device

Use a validated BP monitor (e.g., Omron, Withings) to measure your blood pressure.

### Step 2: Record Video Immediately

Within 1-2 minutes of the reference measurement:
- Place finger on phone camera
- Record for 20-30 seconds
- Keep finger steady and relaxed

### Step 3: Run Calibration

```bash
python predict_bp.py your_video.mp4 --calibrate --sbp XXX --dbp YY
```

**Example:**
```bash
# Reference device showed: 120/80 mmHg, HR 72 BPM
python predict_bp.py calibration_video.mp4 --calibrate --sbp 120 --dbp 80 --hr 72
```

**Output:**
```
✓ Calibration saved!
  SBP offset: -7.8 mmHg (predicted: 127.8 → actual: 120.0)
  DBP offset: +6.0 mmHg (predicted: 69.0 → actual: 80.0)
```

---

## Using Calibrated Predictions

After calibration, all future predictions will automatically apply the learned offsets:

```bash
python predict_bp.py new_video.mp4
```

**Output shows:**
```
==================================================
BLOOD PRESSURE ESTIMATION RESULT
==================================================
Systolic BP (SBP):  119.2 mmHg     ← Calibrated
Diastolic BP (DBP): 74.3 mmHg      ← Calibrated  
Heart Rate:         68.9 BPM
Signal Quality:     Fair (confidence: 0.60)

✓ Calibrated prediction
  (Raw: 127.0/68.3 mmHg)           ← Original model output
==================================================
```

---

## Multiple Calibrations

You can add up to **5 calibration measurements** for better accuracy.

### Why Multiple Calibrations?

- Averages out random variations
- More robust to measurement errors
- Accounts for time-of-day variations

### How It Works

The system uses **weighted averaging**:
- Recent calibrations have higher weight
- Older calibrations gradually fade
- Maximum 5 calibrations stored

**Example workflow:**
```bash
# Calibration 1 (morning)
python predict_bp.py morning1.mp4 --calibrate --sbp 118 --dbp 78

# Calibration 2 (evening)
python predict_bp.py evening1.mp4 --calibrate --sbp 122 --dbp 80

# Calibration 3 (next day)
python predict_bp.py day2.mp4 --calibrate --sbp 120 --dbp 79
```

---

## Managing Calibration

### View Current Calibration

```bash
python predict_bp.py --show-calibration
```

**Output:**
```
Calibration: 3 measurement(s) on file
  Latest: 2025-12-08
  SBP offset: -7.8 mmHg
  DBP offset: +6.0 mmHg
```

### Clear Calibration

To start fresh:
```bash
python predict_bp.py --clear-calibration
```

---

## Best Practices

### ✅ DO

1. **Calibrate in similar conditions** as future measurements
   - Same time of day
   - Similar activity level
   - Similar posture (sitting/standing)

2. **Use high-quality reference device**
   - Clinically validated BP monitor
   - Properly sized cuff
   - Follow manufacturer instructions

3. **Take video immediately after reference**
   - Within 1-2 minutes
   - BP can change quickly

4. **Add multiple calibrations over days**
   - Morning and evening
   - Different days
   - Builds robust average

### ❌ DON'T

1. **Don't calibrate with poor quality video**
   - Ensure good signal quality first
   - Check that `Signal Quality: Good` or `Fair`

2. **Don't mix different conditions**
   - Don't calibrate after exercise, then measure at rest
   - Keep activity levels consistent

3. **Don't rely on single calibration**
   - Use 3-5 measurements for best results

4. **Don't use manual sphygmomanometer values**
   - Human measurement error is too high
   - Use automated devices only

---

## How Calibration Works (Technical)

### Offset Calculation

```
SBP offset = Actual SBP - Predicted SBP
DBP offset = Actual DBP - Predicted DBP
```

**Example:**
- Model predicts: 127.8/69.0 mmHg
- Reference device: 120.0/75.0 mmHg
- Offsets: -7.8 SBP, +6.0 DBP

### Weighted Averaging

With multiple calibrations, the system uses linear weighting:

```
Weight(calibration_i) = i + 1

For 3 calibrations:
- Calibration 1: weight = 1
- Calibration 2: weight = 2  
- Calibration 3: weight = 3 (most recent, highest weight)

Average offset = Σ(offset_i × weight_i) / Σ(weight_i)
```

### Application

```
Calibrated BP = Raw Prediction + Average Offset
```

---

## Storage

Calibration data is stored in:
```
model/calibration.json
```

**Format:**
```json
{
  "calibrations": [
    {
      "timestamp": "2025-12-08T11:30:00",
      "predicted": {"sbp": 127.8, "dbp": 69.0, "hr": 72.5},
      "actual": {"sbp": 120.0, "dbp": 75.0, "hr": 70.0},
      "offsets": {"sbp": -7.8, "dbp": 6.0},
      "scales": {"sbp": 0.939, "dbp": 1.087}
    }
  ]
}
```

---

## Limitations

1. **Not a replacement for medical devices**
   - Still an estimation tool
   - Use for monitoring trends, not diagnosis

2. **Calibration is user-specific**
   - Each person needs their own calibration
   - Don't share calibration files

3. **May drift over time**
   - Re-calibrate monthly
   - Or if you notice systematic errors

4. **Assumes linear offset**
   - Works best for normal BP ranges
   - May be less accurate for extreme values

---

## Troubleshooting

### "Calibrated prediction same as raw"

**Cause:** No calibration data loaded

**Solution:** 
```bash
python predict_bp.py --show-calibration  # Check status
```

### "Large offsets (>20 mmHg)"

**Causes:**
- Poor video quality during calibration
- Reference device error
- Very different conditions

**Solution:**
- Clear calibration and retry
- Ensure good signal quality
- Use validated reference device

### "Predictions still inaccurate after calibration"

**Causes:**
- Only one calibration point
- High variability in BP
- Model limitations

**Solution:**
- Add 3-5 calibrations over several days
- Ensure consistent measurement conditions
- Consider re-training model with more data

---

## Example Workflow

### Initial Setup (Day 1)

```bash
# Morning measurement
python predict_bp.py morning.mp4 --calibrate --sbp 118 --dbp 76
```

### Add More Data (Day 2-3)

```bash
# Evening  
python predict_bp.py evening.mp4 --calibrate --sbp 122 --dbp 80

# Next morning
python predict_bp.py day2_morning.mp4 --calibrate --sbp 119 --dbp 77
```

### Daily Use

```bash
# Now all predictions are calibrated automatically
python predict_bp.py daily_check.mp4

# Output will show: "✓ Calibrated prediction"
```

---

## Scientific Basis

Calibration addresses:
1. **Inter-individual variability**: People have different vascular properties
2. **Sensor differences**: Camera sensors vary (iPhone vs Android)
3. **Physiological factors**: Skin tone, blood perfusion, temperature

Research shows personalized calibration can reduce MAE by 30-50% compared to population-level models.

---

## Questions?

- See `README.md` for general usage
- See `context.md` for technical details
- Check calibration status: `python predict_bp.py --show-calibration`
