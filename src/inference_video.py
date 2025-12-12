"""Video-based Blood Pressure Inference.

This module handles real-world usage: extracting PPG from smartphone videos
and predicting blood pressure using the trained model (RF or CNN).
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

from sklearn.exceptions import InconsistentVersionWarning
import cv2

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import joblib
import numpy as np
import pandas as pd
import torch
from scipy import signal as sp_signal

from features import PPGFeatureExtractor
from model_cnn import load_cnn_model, predict_with_cnn, WINDOW_SIZE


def video_to_bp(
    video_path: str,
    model_path: str = "model/bp_model.pkl",
    model_type: str = "auto",
    calibration_manager=None
) -> dict:
    """Predict blood pressure from a smartphone video.
    
    Args:
        video_path: Path to .mp4 video file
        model_path: Path to trained model file
        model_type: 'rf', 'cnn', or 'auto' (detect from file extension)
        calibration_manager: Optional CalibrationManager
    
    Returns:
        Dictionary with results including SBP, DBP, Heart Rate, etc.
    """
    print(f"Processing video: {video_path}")
    
    # Auto-detect model type
    if model_type == 'auto':
        if 'cnn' in model_path.lower() or model_path.endswith('.pt'):
            model_type = 'cnn'
        else:
            model_type = 'rf'
    
    print(f"Using model: {model_type.upper()} ({model_path})")
    
    # Step 1: Extract PPG signal from video
    print("[1/4] Extracting PPG signal from green channel...")
    ppg_raw, fps = extract_ppg_from_video(video_path)
    
    if ppg_raw is None:
        raise ValueError("Failed to extract PPG signal from video")
    
    print(f"  Extracted {len(ppg_raw)} samples at {fps:.1f} FPS")
    print(f"  Duration: {len(ppg_raw)/fps:.1f} seconds")
    
    # Step 2: Resample to 125 Hz
    print("[2/4] Resampling to 125 Hz...")
    ppg_resampled = resample_to_target(ppg_raw, fps, target_fs=125)
    print(f"  Resampled signal: {len(ppg_resampled)} samples")
    
    # Check signal length
    if len(ppg_resampled) < WINDOW_SIZE:
        raise ValueError(f"Signal too short ({len(ppg_resampled)} < {WINDOW_SIZE} required)")
    
    # Step 3 & 4: Inference (Model specific)
    print(f"[3/4] Running {model_type.upper()} inference...")
    
    if model_type == 'cnn':
        result = _predict_cnn(ppg_resampled, model_path)
    else:
        result = _predict_rf(ppg_resampled, model_path)
        
    sbp_raw, dbp_raw = result['sbp'], result['dbp']
    heart_rate = result['heart_rate']
    quality_score = result.get('confidence', 0.5)

    # Step 5: Calibration
    calibrated = False
    if calibration_manager is not None:
        sbp, dbp = calibration_manager.apply_calibration(sbp_raw, dbp_raw, model_type=model_type)
        # Check if calibration was actually applied (data exists)
        if calibration_manager.calibration_data.get(model_type):
            calibrated = True
    else:
        sbp, dbp = sbp_raw, dbp_raw
    
    output = {
        'sbp': round(sbp, 1),
        'dbp': round(dbp, 1),
        'heart_rate': round(heart_rate, 1),
        'confidence': quality_score,
        'signal_quality': 'Good' if quality_score > 0.7 else 'Fair' if quality_score > 0.4 else 'Poor',
        'calibrated': calibrated,
        'model_type': model_type
    }
    
    if calibrated:
        output['raw_sbp'] = round(sbp_raw, 1)
        output['raw_dbp'] = round(dbp_raw, 1)
    
    _print_results(output)
    
    return output


def _predict_cnn(signal_data: np.ndarray, model_path: str) -> dict:
    """Run inference using CNN model with windowing."""
    # Remove extension if present for load_cnn_model
    if model_path.endswith('.pt'):
        model_path = model_path[:-3]
    
    model, metadata = load_cnn_model(model_path)
    
    # Create overlapping windows
    step = WINDOW_SIZE // 2  # 50% overlap
    windows = []
    
    for i in range(0, len(signal_data) - WINDOW_SIZE + 1, step):
        window = signal_data[i : i + WINDOW_SIZE]
        windows.append(window)
    
    if not windows:
        # Fallback for short signal: pad/truncate
        window = np.resize(signal_data, WINDOW_SIZE)
        windows.append(window)
    
    windows = np.array(windows)  # (N, 625)
    
    # Normalize (CNN expects normalized inputs)
    # Note: Ideally we use the scaler saved with the model
    if 'scaler' in metadata:
        scaler = metadata['scaler']
        windows_norm = scaler.transform(windows)
    else:
        print("⚠ No scaler found in metadata, using per-window normalization")
        windows_norm = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(axis=1, keepdims=True)
    
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_name = "Apple Silicon GPU"
    else:
        device = 'cpu'
        device_name = "CPU"
    
    print(f"  ✓ Device: {device_name}")

    # Predict
    preds = predict_with_cnn(model, windows_norm, device=device)
    
    # Average predictions
    avg_pred = np.mean(preds, axis=0)
    
    # Estimate HR using robust feature extraction (windowing + averaging)
    features = extract_features_from_signal(signal_data, sampling_rate=125)
    hr = features.get('PPG_Rate_Mean', 75) if features else 75
    
    return {
        'sbp': avg_pred[0],
        'dbp': avg_pred[1],
        'heart_rate': hr,
        'confidence': 0.8  # Placeholder for CNN confidence
    }


def _predict_rf(signal_data: np.ndarray, model_path: str) -> dict:
    """Run inference using Random Forest model."""
    # Extract features
    features = extract_features_from_signal(signal_data, sampling_rate=125)
    
    if features is None:
        raise ValueError("Failed to extract features (signal quality too poor)")
    
    model_package = joblib.load(model_path)
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # Prepare features
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=feature_names, fill_value=0)
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Signal quality assessment
    quality_score = assess_signal_quality(signal_data, features)
    
    return {
        'sbp': prediction[0],
        'dbp': prediction[1],
        'heart_rate': features.get('PPG_Rate_Mean', 0),
        'confidence': quality_score
    }


def extract_ppg_from_video(video_path: str) -> tuple[Optional[np.ndarray], float]:
    """Extract PPG signal from video by selecting the best color channel."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    blue_values = []
    green_values = []
    red_values = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate mean intensity (B, G, R)
        means = cv2.mean(frame)[:3]
        blue_values.append(means[0])
        green_values.append(means[1])
        red_values.append(means[2])
    
    cap.release()
    
    if not green_values:
        return None, 0
    
    # Convert to arrays
    signals = {
        'Blue': np.array(blue_values),
        'Green': np.array(green_values),
        'Red': np.array(red_values)
    }
    
    # Select channel with highest variance (best signal quality)
    best_channel = max(signals, key=lambda k: np.var(signals[k]))
    print(f"  Selected {best_channel} channel")
    
    return signals[best_channel], fps


def resample_to_target(signal_data: np.ndarray, original_fps: float, target_fs: int = 125) -> np.ndarray:
    """Resample signal to match training data frequency."""
    duration = len(signal_data) / original_fps
    target_samples = int(duration * target_fs)
    return sp_signal.resample(signal_data, target_samples)


def extract_features_from_signal(ppg_signal: np.ndarray, sampling_rate: int = 125) -> Optional[dict]:
    """Extract and average features from multiple windows."""
    window_size = sampling_rate * 5
    
    # Extract features from multiple windows
    extractor = PPGFeatureExtractor(sampling_rate=sampling_rate, verbose=False)
    num_windows = len(ppg_signal) // window_size
    all_features = []
    
    for i in range(num_windows):
        start = i * window_size
        window = ppg_signal[start : start + window_size]
        features = extractor.process_window(window)
        if features:
            all_features.append(features)
    
    if not all_features:
        return None
    
    # Average features
    avg_features = {}
    for key in all_features[0].keys():
        values = [f[key] for f in all_features if key in f]
        avg_features[key] = np.mean(values)
    
    return avg_features


def assess_signal_quality(ppg_signal: np.ndarray, features: dict) -> float:
    """Assess signal quality based on physiology rules."""
    score = 1.0
    
    # Peak density check
    if 'PPG_NumPeaks' in features:
        expected_peaks = len(ppg_signal) / 125 * 1.5  # ~90 BPM
        ratio = features['PPG_NumPeaks'] / expected_peaks
        if ratio < 0.5: score *= 0.6
    
    # Variability check
    if features.get('PPG_Std', 0) < 1.0: score *= 0.5
    
    # HR range check
    hr = features.get('PPG_Rate_Mean', 75)
    if not (40 <= hr <= 180): score *= 0.4
    
    return min(1.0, max(0.0, score))


def _print_results(result: dict) -> None:
    """Print formatted results."""
    print("\n" + "=" * 50)
    print(f"BLOOD PRESSURE RESULT ({result['model_type'].upper()})")
    print("=" * 50)
    print(f"Systolic BP (SBP):  {result['sbp']:.1f} mmHg")
    print(f"Diastolic BP (DBP): {result['dbp']:.1f} mmHg")
    print(f"Heart Rate:         {result['heart_rate']:.1f} BPM")
    print(f"Signal Quality:     {result['signal_quality']} (score: {result['confidence']:.2f})")
    
    if result['calibrated']:
        print(f"\n✓ Calibrated prediction")
        print(f"  (Raw: {result['raw_sbp']:.1f}/{result['raw_dbp']:.1f} mmHg)")
    print("=" * 50)
