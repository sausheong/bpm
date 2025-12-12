"""Calibration manager for personalized BP predictions.

This module handles storing and applying user-specific calibration data
to improve prediction accuracy.
"""

import json
import os
from typing import Optional, Tuple
from datetime import datetime


class CalibrationManager:
    """Manage calibration data for personalized BP predictions."""
    
    def __init__(self, calibration_file: str = "model/calibration.json"):
        """Initialize calibration manager.
        
        Args:
            calibration_file: Path to store calibration data
        """
        self.calibration_file = calibration_file
        self.calibration_data = self._load_calibration()
    
    def _load_calibration(self) -> dict:
        """Load calibration data from file."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_calibration(self):
        """Save calibration data to file."""
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
    
    def calibrate(self, 
                  predicted_sbp: float, 
                  predicted_dbp: float,
                  predicted_hr: float,
                  actual_sbp: float, 
                  actual_dbp: float,
                  actual_hr: Optional[float] = None) -> dict:
        """Store calibration measurement.
        
        Args:
            predicted_sbp: Model's predicted systolic BP
            predicted_dbp: Model's predicted diastolic BP
            predicted_hr: Model's predicted heart rate (from PPG_Rate_Mean)
            actual_sbp: User's actual systolic BP (from reference device)
            actual_dbp: User's actual diastolic BP (from reference device)
            actual_hr: Optional actual heart rate for additional calibration
        
        Returns:
            Calibration offsets dictionary
        """
        # Calculate offsets
        sbp_offset = actual_sbp - predicted_sbp
        dbp_offset = actual_dbp - predicted_dbp
        
        # Calculate scaling factors (ratio approach)
        sbp_scale = actual_sbp / predicted_sbp if predicted_sbp > 0 else 1.0
        dbp_scale = actual_dbp / predicted_dbp if predicted_dbp > 0 else 1.0
        
        calibration = {
            'timestamp': datetime.now().isoformat(),
            'predicted': {
                'sbp': predicted_sbp,
                'dbp': predicted_dbp,
                'hr': predicted_hr
            },
            'actual': {
                'sbp': actual_sbp,
                'dbp': actual_dbp,
                'hr': actual_hr
            },
            'offsets': {
                'sbp': sbp_offset,
                'dbp': dbp_offset
            },
            'scales': {
                'sbp': sbp_scale,
                'dbp': dbp_scale
            }
        }
        
        # Store calibration (keep last 5 calibrations for averaging)
        if 'calibrations' not in self.calibration_data:
            self.calibration_data['calibrations'] = []
        
        self.calibration_data['calibrations'].append(calibration)
        
        # Keep only last 5
        if len(self.calibration_data['calibrations']) > 5:
            self.calibration_data['calibrations'] = self.calibration_data['calibrations'][-5:]
        
        self._save_calibration()
        
        print(f"\n✓ Calibration saved!")
        print(f"  SBP offset: {sbp_offset:+.1f} mmHg (predicted: {predicted_sbp:.1f} → actual: {actual_sbp:.1f})")
        print(f"  DBP offset: {dbp_offset:+.1f} mmHg (predicted: {predicted_dbp:.1f} → actual: {actual_dbp:.1f})")
        
        return calibration
    
    def apply_calibration(self, predicted_sbp: float, predicted_dbp: float) -> Tuple[float, float]:
        """Apply calibration to new predictions.
        
        Uses weighted average of recent calibrations, with more recent
        calibrations weighted higher.
        
        Args:
            predicted_sbp: Raw model prediction for SBP
            predicted_dbp: Raw model prediction for DBP
        
        Returns:
            (calibrated_sbp, calibrated_dbp)
        """
        if 'calibrations' not in self.calibration_data or not self.calibration_data['calibrations']:
            # No calibration data - return raw predictions
            return predicted_sbp, predicted_dbp
        
        calibrations = self.calibration_data['calibrations']
        n = len(calibrations)
        
        # Use weighted average of offsets (more recent = higher weight)
        sbp_offset_total = 0
        dbp_offset_total = 0
        weight_total = 0
        
        for i, cal in enumerate(calibrations):
            # Weight increases with recency (linear weighting)
            weight = i + 1  # 1, 2, 3, 4, 5
            sbp_offset_total += cal['offsets']['sbp'] * weight
            dbp_offset_total += cal['offsets']['dbp'] * weight
            weight_total += weight
        
        avg_sbp_offset = sbp_offset_total / weight_total
        avg_dbp_offset = dbp_offset_total / weight_total
        
        # Apply offsets
        calibrated_sbp = predicted_sbp + avg_sbp_offset
        calibrated_dbp = predicted_dbp + avg_dbp_offset
        
        return calibrated_sbp, calibrated_dbp
    
    def get_calibration_info(self) -> str:
        """Get human-readable calibration status."""
        if 'calibrations' not in self.calibration_data or not self.calibration_data['calibrations']:
            return "No calibration data available"
        
        n = len(self.calibration_data['calibrations'])
        latest = self.calibration_data['calibrations'][-1]
        
        info = f"Calibration: {n} measurement(s) on file\n"
        info += f"  Latest: {latest['timestamp'][:10]}\n"
        info += f"  SBP offset: {latest['offsets']['sbp']:+.1f} mmHg\n"
        info += f"  DBP offset: {latest['offsets']['dbp']:+.1f} mmHg"
        
        return info
    
    def clear_calibration(self):
        """Clear all calibration data."""
        self.calibration_data = {}
        self._save_calibration()
        print("✓ Calibration data cleared")
