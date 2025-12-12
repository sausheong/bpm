"""UCI Dataset Loader for Blood Pressure Estimation.

This module handles loading .mat files from the UCI Cuff-less Blood Pressure
Estimation dataset and generates windowed samples with BP targets.
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat
import h5py
from typing import Tuple, List, Optional
from tqdm import tqdm
import warnings


class UCILoader:
    """Load UCI Cuff-less BP dataset and generate windowed samples.
    
    The dataset contains synchronized PPG, ABP, and ECG signals sampled at 125 Hz.
    We extract 5-second windows and calculate SBP/DBP targets from ABP peaks/valleys.
    """
    
    SAMPLING_RATE = 125  # Hz
    WINDOW_SECONDS = 5
    WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS  # 625 samples
    BP_MIN, BP_MAX = 60, 180  # Valid BP range in mmHg
    MIN_PEAKS = 3  # Minimum peaks required per window
    
    def __init__(self, verbose: bool = True):
        """Initialize the UCI loader.
        
        Args:
            verbose: If True, show progress bars and warnings
        """
        self.verbose = verbose
    
    def load_part(self, filename: str, max_windows: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load a .mat file and extract windowed PPG samples with BP targets.
        
        Args:
            filename: Path to .mat file (e.g., 'uci_dataset/Part_1.mat')
            max_windows: Maximum number of windows to extract (for debugging)
        
        Returns:
            X_raw: List of PPG windows (each 625 samples)
            y: Array of shape (N, 2) with [SBP, DBP] targets
        """
        if self.verbose:
            print(f"Loading {filename}...")
        
        # Try loading as HDF5 (MATLAB v7.3) first
        try:
            with h5py.File(filename, 'r') as f:
                # Find the main data variable
                data_key = None
                for key in f.keys():
                    if key in ['p', 'data', 'signals'] or not key.startswith('#'):
                        data_key = key
                        break
                
                if data_key is None:
                    raise ValueError(f"Could not find data in HDF5 file. Available keys: {list(f.keys())}")
                
                patient_records = f[data_key]
                
                if self.verbose:
                    print(f"Found {len(patient_records)} patient records (HDF5 format)")
                
                return self._process_hdf5_data(patient_records, max_windows)
        
        except (OSError, KeyError):
            # Fall back to scipy.io.loadmat for older MATLAB formats
            if self.verbose:
                print("Trying older MATLAB format...")
            
            try:
                data = loadmat(filename)
            except Exception as e:
                raise IOError(f"Failed to load {filename}: {e}")
            
            # Find the main data variable
            data_key = None
            for key in ['p', 'data', 'signals']:
                if key in data:
                    data_key = key
                    break
            
            if data_key is None:
                available_keys = [k for k in data.keys() if not k.startswith('__')]
                raise ValueError(f"Could not find data in .mat file. Available keys: {available_keys}")
            
            patient_records = data[data_key]
            
            if self.verbose:
                print(f"Found {len(patient_records)} patient records")
            
            return self._process_legacy_data(patient_records, max_windows)
    
    def _process_hdf5_data(self, patient_records, max_windows: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process HDF5 (MATLAB v7.3) format data."""
        X_raw = []
        y = []
        
        total_windows = 0
        skipped_windows = 0
        
        # Process each patient record
        # Shape is (3000, 1) so we iterate through dimension 0
        n_patients = patient_records.shape[0]
        iterator = tqdm(range(n_patients), desc="Processing patients") if self.verbose else range(n_patients)
        
        for i in iterator:
            try:
                # Get reference to patient data
                patient_ref = patient_records[i, 0]
                
                # Dereference if needed
                if isinstance(patient_ref, h5py.Reference):
                    patient = patient_records.file[patient_ref]
                else:
                    patient = patient_ref
                
                # Extract signals (shape: 3 x N or N x 3)
                signals = np.array(patient)
                
                # Handle transposed data
                if signals.shape[0] > signals.shape[1]:
                    signals = signals.T
                
                if signals.shape[0] < 2:
                    continue
                
                ppg = signals[0, :].flatten()
                abp = signals[1, :].flatten()
                
                # Generate windows from this patient
                patient_X, patient_y, skipped = self._window_signals(ppg, abp)
                
                X_raw.extend(patient_X)
                y.extend(patient_y)
                
                total_windows += len(patient_X) + skipped
                skipped_windows += skipped
                
                # Check if we've reached max_windows
                if max_windows and len(X_raw) >= max_windows:
                    X_raw = X_raw[:max_windows]
                    y = y[:max_windows]
                    break
                    
            except Exception as e:
                if self.verbose and np.random.random() < 0.01:  # Print 1% of errors
                    print(f"\nSkipping patient {i}: {str(e)[:80]}")
                continue
        
        if self.verbose:
            print(f"\nTotal windows extracted: {len(X_raw)}")
            print(f"Skipped windows (quality issues): {skipped_windows}")
            if total_windows > 0:
                print(f"Success rate: {len(X_raw)/total_windows*100:.1f}%")
        
        return X_raw, np.array(y)
    
    def _process_legacy_data(self, patient_records, max_windows: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process legacy MATLAB format data."""
        X_raw = []
        y = []
        
        total_windows = 0
        skipped_windows = 0
        
        # Process each patient record
        iterator = tqdm(patient_records, desc="Processing patients") if self.verbose else patient_records
        
        for patient in iterator:
            # Extract signals (shape: 3 x N)
            if patient.shape[0] < 2:
                continue
            
            ppg = patient[0, :].flatten()
            abp = patient[1, :].flatten()
            
            # Generate windows from this patient
            patient_X, patient_y, skipped = self._window_signals(ppg, abp)
            
            X_raw.extend(patient_X)
            y.extend(patient_y)
            
            total_windows += len(patient_X) + skipped
            skipped_windows += skipped
            
            # Check if we've reached max_windows
            if max_windows and len(X_raw) >= max_windows:
                X_raw = X_raw[:max_windows]
                y = y[:max_windows]
                break
        
        if self.verbose:
            print(f"\nTotal windows extracted: {len(X_raw)}")
            print(f"Skipped windows (quality issues): {skipped_windows}")
            print(f"Success rate: {len(X_raw)/total_windows*100:.1f}%")
        
        return X_raw, np.array(y)

    
    def _window_signals(self, ppg: np.ndarray, abp: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]], int]:
        """Slice continuous signals into 5-second non-overlapping windows.
        
        Args:
            ppg: Full PPG signal for one patient
            abp: Full ABP signal for one patient
        
        Returns:
            ppg_windows: List of PPG windows
            bp_targets: List of [SBP, DBP] pairs
            skipped: Number of windows that were skipped
        """
        ppg_windows = []
        bp_targets = []
        skipped = 0
        
        num_windows = len(ppg) // self.WINDOW_SIZE
        
        for i in range(num_windows):
            start_idx = i * self.WINDOW_SIZE
            end_idx = start_idx + self.WINDOW_SIZE
            
            ppg_window = ppg[start_idx:end_idx]
            abp_window = abp[start_idx:end_idx]
            
            # Extract BP targets from ABP window
            bp_values = self._extract_bp_targets(abp_window)
            
            if bp_values is not None:
                sbp, dbp = bp_values
                ppg_windows.append(ppg_window)
                bp_targets.append([sbp, dbp])
            else:
                skipped += 1
        
        return ppg_windows, bp_targets, skipped
    
    def _extract_bp_targets(self, abp_window: np.ndarray) -> Optional[Tuple[float, float]]:
        """Extract SBP and DBP from ABP waveform.
        
        The ABP signal is a continuous waveform showing arterial pressure:
        - Peaks = Systolic pressure (max arterial pressure during heart contraction)
        - Valleys = Diastolic pressure (min arterial pressure during heart relaxation)
        
        Algorithm:
        1. Detect peaks using scipy.signal.find_peaks with prominence threshold
        2. Detect valleys by finding peaks in the inverted signal
        3. Calculate mean SBP from peaks, mean DBP from valleys
        4. Validate: discard if outside realistic BP range [60, 180] mmHg
        
        Args:
            abp_window: ABP signal window (625 samples)
        
        Returns:
            (mean_sbp, mean_dbp) or None if validation fails
        """
        # Check for flat or invalid signal
        if np.std(abp_window) < 1.0:
            return None
        
        # Detect systolic peaks (peaks in ABP signal)
        # Prominence ensures we only detect actual peaks, not noise
        peaks, _ = signal.find_peaks(abp_window, prominence=10, distance=50)
        
        if len(peaks) < self.MIN_PEAKS:
            return None
        
        # Detect diastolic valleys (peaks in inverted signal)
        valleys, _ = signal.find_peaks(-abp_window, prominence=10, distance=50)
        
        if len(valleys) < self.MIN_PEAKS:
            return None
        
        # Calculate mean SBP and DBP
        sbp = np.mean(abp_window[peaks])
        dbp = np.mean(abp_window[valleys])
        
        # Validate BP values are in realistic range
        if not (self.BP_MIN <= sbp <= self.BP_MAX):
            return None
        if not (self.BP_MIN <= dbp <= self.BP_MAX):
            return None
        
        # Validate that SBP > DBP (basic physiology check)
        if sbp <= dbp:
            return None
        
        return sbp, dbp


if __name__ == "__main__":
    # Quick test
    loader = UCILoader(verbose=True)
    X, y = loader.load_part("uci_dataset/Part_1.mat", max_windows=10)
    print(f"\nTest results:")
    print(f"X shape: {len(X)} windows of {len(X[0])} samples each")
    print(f"y shape: {y.shape}")
    print(f"Sample BP values: {y[:5]}")
