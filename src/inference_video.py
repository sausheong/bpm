"""Video-based Blood Pressure Inference.

This module handles real-world usage: extracting PPG from smartphone videos
and predicting blood pressure using the trained model.
"""

import numpy as np
import cv2
from scipy import signal
import joblib
import pandas as pd
from typing import Tuple, Optional
import warnings

from features import PPGFeatureExtractor


def video_to_bp(video_path: str, model_path: str = "model/bp_model.pkl", calibration_manager=None) -> dict:
    """Predict blood pressure from a smartphone video.
    
    Expected video format:
    - User places finger on phone camera for 20-30 seconds
    - Camera captures changes in blood volume via green channel
    
    Args:
        video_path: Path to .mp4 video file
        model_path: Path to trained model pickle
        calibration_manager: Optional CalibrationManager for personalized predictions
    
    Returns:
        Dictionary with keys:
            - 'sbp': Systolic blood pressure (mmHg)
            - 'dbp': Diastolic blood pressure (mmHg)
            - 'heart_rate': Heart rate (BPM)
            - 'confidence': Quality score (0-1)
            - 'signal_quality': Assessment of PPG signal quality
            - 'raw_sbp': Uncalibrated SBP (if calibration applied)
            - 'raw_dbp': Uncalibrated DBP (if calibration applied)
    """
    print(f"Processing video: {video_path}")
    
    # Step 1: Extract PPG signal from video
    print("[1/4] Extracting PPG signal from green channel...")
    ppg_raw, fps = extract_ppg_from_video(video_path)
    
    if ppg_raw is None:
        raise ValueError("Failed to extract PPG signal from video")
    
    print(f"  Extracted {len(ppg_raw)} samples at {fps:.1f} FPS")
    print(f"  Duration: {len(ppg_raw)/fps:.1f} seconds")
    
    # Step 2: Resample to 125 Hz (match training data)
    print("[2/4] Resampling to 125 Hz...")
    ppg_resampled = resample_to_target(ppg_raw, fps, target_fs=125)
    print(f"  Resampled signal: {len(ppg_resampled)} samples")
    
    # Step 3: Extract features
    print("[3/4] Extracting features...")
    features = extract_features_from_signal(ppg_resampled, sampling_rate=125)
    
    if features is None:
        raise ValueError("Failed to extract features (signal quality too poor)")
    
    print(f"  Extracted {len(features)} features")
    
    # Extract heart rate from features
    heart_rate = features.get('PPG_Rate_Mean', 0)
    
    # Step 4: Load model and predict
    print("[4/4] Loading model and predicting...")
    model_package = joblib.load(model_path)
    
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # Align features with training columns
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=feature_names, fill_value=0)
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    sbp_raw, dbp_raw = prediction
    
    # Apply calibration if available
    if calibration_manager is not None:
        sbp, dbp = calibration_manager.apply_calibration(sbp_raw, dbp_raw)
        calibrated = True
    else:
        sbp, dbp = sbp_raw, dbp_raw
        calibrated = False
    
    # Assess signal quality
    quality_score = assess_signal_quality(ppg_resampled, features)
    
    result = {
        'sbp': round(sbp, 1),
        'dbp': round(dbp, 1),
        'heart_rate': round(heart_rate, 1),
        'confidence': quality_score,
        'signal_quality': 'Good' if quality_score > 0.7 else 'Fair' if quality_score > 0.4 else 'Poor',
        'calibrated': calibrated
    }
    
    # Include raw predictions if calibrated
    if calibrated:
        result['raw_sbp'] = round(sbp_raw, 1)
        result['raw_dbp'] = round(dbp_raw, 1)
    
    print("\n" + "=" * 50)
    print("BLOOD PRESSURE ESTIMATION RESULT")
    print("=" * 50)
    print(f"Systolic BP (SBP):  {result['sbp']:.1f} mmHg")
    print(f"Diastolic BP (DBP): {result['dbp']:.1f} mmHg")
    print(f"Heart Rate:         {result['heart_rate']:.1f} BPM")
    print(f"Signal Quality:     {result['signal_quality']} (confidence: {result['confidence']:.2f})")
    if calibrated:
        print(f"\n✓ Calibrated prediction")
        print(f"  (Raw: {result['raw_sbp']:.1f}/{result['raw_dbp']:.1f} mmHg)")
    print("=" * 50)
    
    return result


def extract_ppg_from_video(video_path: str) -> Tuple[Optional[np.ndarray], float]:
    """Extract PPG signal from video by selecting the best color channel.
    
    Why Color Channels for PPG?
    - Hemoglobin absorbs light differently at different wavelengths
    - Green (~540nm) typically provides best signal
    - However, some videos may have better signal in red or blue channels
    
    Algorithm:
    1. Read video frame by frame
    2. For each frame: extract mean intensity of all three channels
    3. Choose the channel with highest variance (best signal)
    4. Return time series + original FPS
    
    Args:
        video_path: Path to video file
    
    Returns:
        (signal, fps): PPG signal array and video framerate
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    blue_values = []
    green_values = []
    red_values = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract all channels (OpenCV uses BGR format)
        blue_channel = frame[:, :, 0]
        green_channel = frame[:, :, 1]
        red_channel = frame[:, :, 2]
        
        # Calculate mean intensity for each channel
        blue_values.append(np.mean(blue_channel))
        green_values.append(np.mean(green_channel))
        red_values.append(np.mean(red_channel))
        
        frame_count += 1
    
    cap.release()
    
    if len(green_values) == 0:
        return None, 0
    
    # Convert to arrays
    blue_signal = np.array(blue_values)
    green_signal = np.array(green_values)
    red_signal = np.array(red_values)
    
    # Select channel with highest variance (best signal quality)
    blue_var = np.var(blue_signal)
    green_var = np.var(green_signal)
    red_var = np.var(red_signal)
    
    if green_var >= red_var and green_var >= blue_var:
        selected_signal = green_signal
        channel_name = "Green"
    elif red_var >= blue_var:
        selected_signal = red_signal
        channel_name = "Red"
    else:
        selected_signal = blue_signal
        channel_name = "Blue"
    
    print(f"  Selected {channel_name} channel (variance: {np.var(selected_signal):.2f})")
    
    return selected_signal, fps


def resample_to_target(signal_data: np.ndarray, original_fps: float, target_fs: int = 125) -> np.ndarray:
    """Resample signal to match training data frequency.
    
    This is critical because the model was trained on 125 Hz data,
    so inference data must be at the same sampling rate.
    
    Example:
    - Video at 30 FPS, 10 seconds → 300 samples
    - Target: 125 Hz, 10 seconds → 1250 samples
    
    Uses scipy.signal.resample for anti-aliased resampling.
    
    Args:
        signal_data: Original signal array
        original_fps: Original sampling rate
        target_fs: Target sampling rate (default: 125 Hz)
    
    Returns:
        Resampled signal
    """
    duration = len(signal_data) / original_fps
    target_samples = int(duration * target_fs)
    
    # Use scipy.signal.resample for high-quality resampling
    from scipy import signal as sp_signal
    resampled = sp_signal.resample(signal_data, target_samples)
    
    return resampled


def extract_features_from_signal(ppg_signal: np.ndarray, sampling_rate: int = 125) -> Optional[dict]:
    """Extract features from a PPG signal.
    
    For long signals (20-30 seconds), we extract features from multiple
    windows and average them for robustness.
    
    Args:
        ppg_signal: PPG signal array
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of features, or None if extraction fails
    """
    window_size = sampling_rate * 5  # 5-second windows
    
    if len(ppg_signal) < window_size:
        print(f"Warning: Signal too short ({len(ppg_signal)} samples < {window_size} required)")
        return None
    
    # Extract features from multiple windows and average
    extractor = PPGFeatureExtractor(sampling_rate=sampling_rate, verbose=False)
    
    num_windows = len(ppg_signal) // window_size
    all_features = []
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = ppg_signal[start:end]
        
        features = extractor.process_window(window)
        if features is not None:
            all_features.append(features)
    
    if len(all_features) == 0:
        return None
    
    # Average features across all windows
    avg_features = {}
    feature_keys = all_features[0].keys()
    
    for key in feature_keys:
        values = [f[key] for f in all_features if key in f]
        avg_features[key] = np.mean(values)
    
    return avg_features


def assess_signal_quality(ppg_signal: np.ndarray, features: dict) -> float:
    """Assess the quality of the PPG signal.
    
    Returns a confidence score between 0 and 1 based on:
    - Signal-to-noise ratio
    - Number of detected peaks
    - Regularity of heartbeat
    
    Args:
        ppg_signal: Raw PPG signal
        features: Extracted features
    
    Returns:
        Quality score (0-1)
    """
    score = 1.0
    
    # Check if enough peaks were detected
    if 'PPG_NumPeaks' in features:
        expected_peaks = len(ppg_signal) / 125 * 1.5  # Roughly 90 bpm
        peak_ratio = features['PPG_NumPeaks'] / expected_peaks
        if peak_ratio < 0.5:
            score *= 0.6
    
    # Check signal variability (too flat = poor contact)
    if 'PPG_Std' in features and 'PPG_Range' in features:
        if features['PPG_Std'] < 1.0 or features['PPG_Range'] < 5.0:
            score *= 0.5
    
    # Check for reasonable heart rate variability
    if 'PPG_Rate_Mean' in features:
        hr = features['PPG_Rate_Mean']
        if hr < 40 or hr > 180:  # Unrealistic heart rate
            score *= 0.4
    
    return min(1.0, max(0.0, score))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict BP from video")
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='model/bp_model.pkl',
                        help='Path to trained model')
    
    args = parser.parse_args()
    
    result = video_to_bp(args.video, args.model)
    
    print("\nResult:")
    print(f"  BP: {result['sbp']}/{result['dbp']} mmHg")
    print(f"  Quality: {result['signal_quality']}")
