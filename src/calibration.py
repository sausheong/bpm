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
                    data = json.load(f)
                    # Migrate legacy format if necessary
                    if 'calibrations' in data and 'rf' not in data:
                        return {
                            'rf': data['calibrations'],
                            'cnn': []
                        }
                    return data
            except:
                return {'rf': [], 'cnn': []}
        return {'rf': [], 'cnn': []}
    
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
                  model_type: str = 'rf',
                  actual_hr: Optional[float] = None) -> dict:
        """Store calibration measurement.
        
        Args:
            predicted_sbp: Model's predicted systolic BP
            predicted_dbp: Model's predicted diastolic BP
            predicted_hr: Model's predicted heart rate (from PPG_Rate_Mean)
            actual_sbp: User's actual systolic BP (from reference device)
            actual_dbp: User's actual diastolic BP (from reference device)
            model_type: 'rf' or 'cnn'
            actual_hr: Optional actual heart rate for additional calibration
        
        Returns:
            Calibration offsets dictionary
        """
        model_type = model_type.lower()
        if model_type not in ['rf', 'cnn']:
            model_type = 'rf'  # Default fallback
            
        print(f"Calibrating for model: {model_type.upper()}")

        # Cast to float for JSON serialization
        predicted_sbp = float(predicted_sbp)
        predicted_dbp = float(predicted_dbp)
        predicted_hr = float(predicted_hr)
        actual_sbp = float(actual_sbp)
        actual_dbp = float(actual_dbp)
        if actual_hr is not None:
            actual_hr = float(actual_hr)

        # Calculate offsets
        sbp_offset = actual_sbp - predicted_sbp
        dbp_offset = actual_dbp - predicted_dbp
        
        # Calculate scaling factors (ratio approach)
        sbp_scale = actual_sbp / predicted_sbp if predicted_sbp > 0 else 1.0
        dbp_scale = actual_dbp / predicted_dbp if predicted_dbp > 0 else 1.0
        
        calibration = {
            'timestamp': datetime.now().isoformat(),
            'model': model_type,
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
        
        # Ensure key exists
        if model_type not in self.calibration_data:
            self.calibration_data[model_type] = []
            
        self.calibration_data[model_type].append(calibration)
        
        # Keep only last 5 per model
        if len(self.calibration_data[model_type]) > 5:
            self.calibration_data[model_type] = self.calibration_data[model_type][-5:]
        
        self._save_calibration()
        
        print(f"\n✓ Calibration saved for {model_type.upper()}!")
        print(f"  SBP offset: {sbp_offset:+.1f} mmHg (predicted: {predicted_sbp:.1f} → actual: {actual_sbp:.1f})")
        print(f"  DBP offset: {dbp_offset:+.1f} mmHg (predicted: {predicted_dbp:.1f} → actual: {actual_dbp:.1f})")
        
        return calibration
    
    def apply_calibration(self, 
                         predicted_sbp: float, 
                         predicted_dbp: float,
                         model_type: str = 'rf') -> Tuple[float, float]:
        """Apply calibration to new predictions.
        
        Uses weighted average of recent calibrations for the specific model.
        
        Args:
            predicted_sbp: Raw model prediction for SBP
            predicted_dbp: Raw model prediction for DBP
            model_type: 'rf' or 'cnn'
        
        Returns:
            (calibrated_sbp, calibrated_dbp)
        """
        model_type = model_type.lower()
        
        # Fallback for auto/unknown types or empty lists
        if model_type not in self.calibration_data or not self.calibration_data.get(model_type):
            return predicted_sbp, predicted_dbp
        
        calibrations = self.calibration_data[model_type]
        
        # Use weighted average of offsets (more recent = higher weight)
        sbp_offset_total = 0
        dbp_offset_total = 0
        weight_total = 0
        
        for i, cal in enumerate(calibrations):
            # Weight increases with recency
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
        """Get human-readable calibration status for all models."""
        info = []
        
        # Handle legacy or non-dict structure (safety check)
        if hasattr(self.calibration_data, 'get'):
            rf_cals = self.calibration_data.get('rf', [])
            cnn_cals = self.calibration_data.get('cnn', [])
        else:
            return "Invalid calibration data format"

        if not rf_cals and not cnn_cals:
            return "No calibration data available"

        if rf_cals:
            latest = rf_cals[-1]
            info.append(f"RF Model: {len(rf_cals)} measurement(s)")
            info.append(f"  Latest: {latest['timestamp'][:16]}")
            info.append(f"  Offset: SBP {latest['offsets']['sbp']:+.1f}, DBP {latest['offsets']['dbp']:+.1f}")
            info.append("")

        if cnn_cals:
            latest = cnn_cals[-1]
            info.append(f"CNN Model: {len(cnn_cals)} measurement(s)")
            info.append(f"  Latest: {latest['timestamp'][:16]}")
            info.append(f"  Offset: SBP {latest['offsets']['sbp']:+.1f}, DBP {latest['offsets']['dbp']:+.1f}")

        return "\n".join(info)
    
    def clear_calibration(self):
        """Clear all calibration data."""
        self.calibration_data = {'rf': [], 'cnn': []}
        self._save_calibration()
        print("✓ Calibration data cleared")
