#!/usr/bin/env python3
"""Blood Pressure Prediction from Video

Script to predict blood pressure from a video file with optional calibration.
Supports both Random Forest (RF) and CNN models.

Usage:
    # Use default model (Random Forest)
    python predict.py video.mp4
    
    # Use CNN model explicitly
    python predict.py video.mp4 --model cnn
    
    # Use specific model file
    python predict.py video.mp4 --model model/my_custom_model.pt
    
    # Calibrate
    python predict.py video.mp4 --calibrate --sbp 120 --dbp 80
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference_video import video_to_bp
from src.calibration import CalibrationManager


def main():
    parser = argparse.ArgumentParser(
        description='Predict BP from video (RF or CNN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random Forest prediction (Default)
  python predict.py video.mp4 --model rf
  
  # CNN prediction (Recommended for accuracy)
  python predict.py video.mp4 --model cnn
  
  # Calibrate with reference measurement
  python predict.py video.mp4 --calibrate --sbp 120 --dbp 80
        """
    )
    
    parser.add_argument('video', nargs='?', type=str, help='Path to video file')
    
    parser.add_argument(
        '--model', type=str, default='rf',
        help='Model type ("rf", "cnn") or path to model file'
    )
    
    # Calibration options
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibration mode: save reference BP measurement')
    parser.add_argument('--sbp', type=float,
                        help='Reference systolic BP (mmHg)')
    parser.add_argument('--dbp', type=float,
                        help='Reference diastolic BP (mmHg)')
    parser.add_argument('--hr', type=float,
                        help='Reference heart rate (BPM) (optional)')
    
    # Calibration management
    parser.add_argument('--show-calibration', action='store_true',
                        help='Show current calibration status')
    parser.add_argument('--clear-calibration', action='store_true',
                        help='Clear all calibration data')
    
    args = parser.parse_args()
    
    # Initialize calibration manager
    cal_manager = CalibrationManager()
    
    # Handle calibration management commands
    if args.clear_calibration:
        cal_manager.clear_calibration()
        return
    
    if args.show_calibration:
        print(cal_manager.get_calibration_info())
        return
    
    # Require video file for prediction
    if not args.video:
        parser.print_help()
        return
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Determine model path and type
    model_input = args.model.lower()
    
    if model_input == 'rf':
        model_path = 'model/bp_model.pkl'
        model_type = 'rf'
    elif model_input == 'cnn':
        model_path = 'model/bp_model_cnn'  # Will load .pt and metadata
        model_type = 'cnn'
    else:
        # Custom path provided
        model_path = args.model
        model_type = 'auto'
    
    # Check if model exists (handle .pt extension for CNN)
    if model_type == 'cnn' or (model_type == 'auto' and '.pt' in model_path):
        if not os.path.exists(f"{model_path}.pt") and not os.path.exists(model_path):
             print(f"Error: CNN model not found at {model_path}.pt")
             print("Please train it first: python train.py --model cnn")
             sys.exit(1)
    elif not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train it first: python train.py --model rf")
        sys.exit(1)
    
    try:
        # Make prediction
        result = video_to_bp(
            args.video,
            model_path=model_path,
            model_type=model_type,
            calibration_manager=cal_manager
        )
        
        # Handle Calibration Saving
        if args.calibrate:
            if args.sbp is None or args.dbp is None:
                print("\n⚠ Calibration requires --sbp and --dbp values")
                sys.exit(1)
            
            # Use raw predictions for calibration base
            raw_sbp = result.get('raw_sbp', result['sbp'])
            raw_dbp = result.get('raw_dbp', result['dbp'])
            
            cal_manager.calibrate(
                predicted_sbp=raw_sbp,
                predicted_dbp=raw_dbp,
                predicted_hr=result['heart_rate'],
                actual_sbp=args.sbp,
                actual_dbp=args.dbp,
                actual_hr=args.hr
            )
            
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        # Only show traceback in debug mode or if requested (keeping it concise for user)
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
