"""PPG Feature Extraction using NeuroKit2.

This module extracts morphological features from PPG signals that correlate
with blood pressure. Features include heart rate variability, pulse morphology,
and timing characteristics.
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings


class PPGFeatureExtractor:
    """Extract morphological features from PPG signals using NeuroKit2.
    
    The feature extraction pipeline:
    1. Clean signal (bandpass filter 0.5-8 Hz)
    2. Detect peaks (systolic peaks in PPG)
    3. Extract morphological and HRV features
    """
    
    def __init__(self, sampling_rate: int = 125, verbose: bool = True):
        """Initialize the feature extractor.
        
        Args:
            sampling_rate: Sampling frequency in Hz (default: 125 Hz for UCI data)
            verbose: If True, show progress and warnings
        """
        self.sampling_rate = sampling_rate
        self.verbose = verbose
    
    def process_window(self, ppg_signal: np.ndarray) -> Optional[dict]:
        """Process a single PPG window and extract features.
        
        Steps:
        1. Normalize signal to improve peak detection
        2. Clean signal: nk.ppg_clean(signal, sampling_rate)
           - Applies bandpass filter (0.5-8 Hz by default)
           - Removes baseline wander and high-frequency noise
        
        3. Find peaks: nk.ppg_findpeaks(cleaned, sampling_rate)
           - Uses derivative-based peak detection
           - Returns indices of systolic peaks
        
        4. Analyze: nk.ppg_analyze(peaks, sampling_rate)
           - Extracts: PPG rate, HRV metrics, morphology features
        
        Args:
            ppg_signal: 1D numpy array of PPG samples (typically 625 for 5s window)
        
        Returns:
            Dictionary of features, or None if extraction fails
        """
        try:
            # Check for valid signal
            if len(ppg_signal) < 100:
                return None
            
            if np.std(ppg_signal) < 0.01:  # Too flat
                return None
            
            # Normalize the signal to have better amplitude for NeuroKit2
            # Many PPG signals from the dataset have small amplitudes
            ppg_normalized = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
            ppg_normalized = ppg_normalized * 100  # Scale to reasonable amplitude
            
            # Suppress NeuroKit warnings for noisy signals
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Step 1: Clean the PPG signal
                cleaned = nk.ppg_clean(ppg_normalized, sampling_rate=self.sampling_rate)
                
                # Step 2: Find peaks
                # Note: in NeuroKit2 0.2.x, ppg_findpeaks returns just a dict, not a tuple
                try:
                    info = nk.ppg_findpeaks(cleaned, sampling_rate=self.sampling_rate)
                except Exception:
                    # If default method fails, try alternative method
                    try:
                        info = nk.ppg_findpeaks(cleaned, sampling_rate=self.sampling_rate, method="elgendi")
                    except:
                        return None
                
                # Check if enough peaks were detected
                peaks = info['PPG_Peaks']
                if len(peaks) < 3:
                    return None
                
                # Step 3: Process the PPG signal with peaks
                # Use ppg_process for complete analysis
                signals, info_full = nk.ppg_process(cleaned, sampling_rate=self.sampling_rate)
                
                # Basic features from signals
                features_dict = {}
                
                # Heart rate from peaks
                peak_intervals = np.diff(peaks) / self.sampling_rate  # in seconds
                if len(peak_intervals) > 0:
                    hr_mean = 60.0 / np.mean(peak_intervals)  # beats per minute
                    features_dict['PPG_Rate_Mean'] = hr_mean
                    features_dict['PPG_Rate_Std'] = np.std(60.0 / peak_intervals) if len(peak_intervals) > 1 else 0
                
                # HRV time-domain features
                if len(peak_intervals) > 1:
                    features_dict['HRV_MeanNN'] = np.mean(peak_intervals) * 1000  # in ms
                    features_dict['HRV_SDNN'] = np.std(peak_intervals) * 1000
                    if len(peak_intervals) > 2:
                        successive_diffs = np.diff(peak_intervals)
                        features_dict['HRV_RMSSD'] = np.sqrt(np.mean(successive_diffs**2)) * 1000
                
                # Pulse amplitude features (from cleaned signal)
                peak_amplitudes = cleaned[peaks]
                features_dict['PPG_Peaks_Amplitude_Mean'] = np.mean(peak_amplitudes)
                features_dict['PPG_Peaks_Amplitude_Std'] = np.std(peak_amplitudes)
                
                # Add some basic signal statistics (use original signal stats)
                features_dict['PPG_Mean'] = np.mean(ppg_signal)
                features_dict['PPG_Std'] = np.std(ppg_signal)
                features_dict['PPG_Range'] = np.ptp(ppg_signal)
                features_dict['PPG_NumPeaks'] = len(peaks)
                
                # Filter out NaN and inf values
                features_dict = {k: v for k, v in features_dict.items() 
                                if not (isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)))}
                
                return features_dict
                
        except Exception as e:
            if self.verbose:
                # Only print a few times to avoid spam
                if np.random.random() < 0.01:  # 1% chance
                    print(f"Feature extraction failed: {str(e)[:50]}")
            return None
    
    def transform(self, X_raw_list: List[np.ndarray], n_workers: Optional[int] = None) -> Tuple[pd.DataFrame, List[int]]:
        """Process multiple windows in parallel.
        
        Uses multiprocessing to speed up feature extraction for large datasets.
        Returns both the features and the indices of successfully processed windows
        (to align with y labels).
        
        Args:
            X_raw_list: List of raw PPG windows (each typically 625 samples)
            n_workers: Number of parallel workers (default: CPU count - 1)
        
        Returns:
            features_df: DataFrame with feature columns
            valid_indices: List of indices where feature extraction succeeded
        """
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        
        if self.verbose:
            print(f"Extracting features from {len(X_raw_list)} windows using {n_workers} workers...")
        
        # Process windows in parallel
        with Pool(n_workers) as pool:
            if self.verbose:
                results = list(tqdm(
                    pool.imap(self.process_window, X_raw_list),
                    total=len(X_raw_list),
                    desc="Extracting features"
                ))
            else:
                results = pool.map(self.process_window, X_raw_list)
        
        # Separate successful extractions from failures
        valid_indices = []
        feature_dicts = []
        
        for idx, result in enumerate(results):
            if result is not None:
                valid_indices.append(idx)
                feature_dicts.append(result)
        
        if len(feature_dicts) == 0:
            raise ValueError("Feature extraction failed for all windows!")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_dicts)
        
        # Handle any NaN or infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        if self.verbose:
            print(f"Successfully extracted features from {len(valid_indices)}/{len(X_raw_list)} windows")
            print(f"Feature matrix shape: {features_df.shape}")
            print(f"Features: {list(features_df.columns[:10])}{'...' if len(features_df.columns) > 10 else ''}")
        
        return features_df, valid_indices


def _process_single(args) -> Optional[dict]:
    """Helper for multiprocessing (needs to be at module level)."""
    signal_data, sampling_rate = args
    extractor = PPGFeatureExtractor(sampling_rate=sampling_rate, verbose=False)
    return extractor.process_window(signal_data)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing PPG feature extraction...")
    
    # Generate synthetic PPG signal (5 seconds at 125 Hz)
    t = np.linspace(0, 5, 625)
    # Simulate heartbeat at 75 bpm (1.25 Hz)
    synthetic_ppg = np.sin(2 * np.pi * 1.25 * t) + 0.3 * np.sin(2 * np.pi * 2.5 * t)
    synthetic_ppg += np.random.normal(0, 0.1, len(t))
    
    extractor = PPGFeatureExtractor(sampling_rate=125)
    features = extractor.process_window(synthetic_ppg)
    
    if features:
        print(f"Extracted {len(features)} features")
        print("Sample features:", list(features.items())[:5])
    else:
        print("Feature extraction failed on synthetic signal")
