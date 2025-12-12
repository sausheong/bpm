#!/usr/bin/env python3
"""Blood Pressure Prediction from Video

Script to predict blood pressure from a video file with optional calibration.

Usage:
    # Normal prediction
    python predict.py video.mp4
    
    # Calibration mode (provide reference BP from a real device)
    python predict.py video.mp4 --calibrate --sbp 120 --dbp 80 --hr 72
    
    # View calibration status
    python predict.py --show-calibration
    
    # Clear calibration
    python predict.py --clear-calibration
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inference_video import video_to_bp
from src.calibration import CalibrationManager


def main():
    parser = argparse.ArgumentParser(
        description='Predict BP from video with optional calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard prediction
  python predict.py video.mp4
  
  # Calibrate with reference measurement
  python predict.py video.mp4 --calibrate --sbp 120 --dbp 80
  
  # Include pulse rate for calibration
  python predict.py video.mp4 --calibrate --sbp 120 --dbp 80 --hr 72
        """
    )
    
    parser.add_argument('video', nargs='?', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='model/bp_model.pkl',
                        help='Path to trained model')
    
    # Calibration options
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibration mode: save reference BP measurement')
    parser.add_argument('--sbp', type=float,
                        help='Reference systolic BP (mmHg) for calibration')
    parser.add_argument('--dbp', type=float,
                        help='Reference diastolic BP (mmHg) for calibration')
    parser.add_argument('--hr', type=float,
                        help='Reference heart rate (BPM) for calibration (optional)')
    
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
    
    video_path = args.video
    model_path = args.model
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Please train the model first using: python train.py")
        sys.exit(1)
    
    try:
        # Make prediction
        result = video_to_bp(video_path, model_path, calibration_manager=cal_manager)
        
        # Calibration mode: save reference measurement
        if args.calibrate:
            if args.sbp is None or args.dbp is None:
                print("\n⚠ Calibration requires --sbp and --dbp values")
                print("Example: python predict.py video.mp4 --calibrate --sbp 120 --dbp 80")
                sys.exit(1)
            
            # Use raw predictions for calibration (before any calibration was applied)
            predicted_sbp = result.get('raw_sbp', result['sbp'])
            predicted_dbp = result.get('raw_dbp', result['dbp'])
            
            cal_manager.calibrate(
                predicted_sbp=predicted_sbp,
                predicted_dbp=predicted_dbp,
                predicted_hr=result['heart_rate'],
                actual_sbp=args.sbp,
                actual_dbp=args.dbp,
                actual_hr=args.hr
            )
            
            print("\n" + "=" * 60)
            print("CALIBRATION COMPLETE")
            print("=" * 60)
            print("Future predictions will be adjusted based on this reference.")
            print("You can add up to 5 calibration measurements for better accuracy.")
            print("=" * 60)
            
        else:
            # Normal prediction - show recommendations
            print("\n" + "=" * 60)
            print("RECOMMENDATION")
            print("=" * 60)
            
            if result['signal_quality'] == 'Poor':
                print("⚠ Signal quality is poor. For better results:")
                print("  - Ensure finger fully covers camera")
                print("  - Hold steady for 20-30 seconds")
                print("  - Ensure good lighting")
                print("  - Relax and breathe normally")
            else:
                sbp, dbp = result['sbp'], result['dbp']
                hr = result['heart_rate']
                
                # BP assessment
                if sbp >= 140 or dbp >= 90:
                    print("⚠ Blood pressure is elevated (Hypertensive range)")
                    print("  Consider consulting a healthcare provider")
                elif sbp >= 120 or dbp >= 80:
                    print("⚡ Blood pressure is in the elevated range")
                    print("  Monitor regularly and maintain healthy lifestyle")
                else:
                    print("✓ Blood pressure is in the normal range")
                
                # Heart rate assessment
                print(f"\nHeart Rate: {hr:.0f} BPM ", end="")
                if 60 <= hr <= 100:
                    print("(Normal range)")
                elif hr < 60:
                    print("(Below normal - Bradycardia)")
                else:
                    print("(Above normal - Tachycardia)")
            
            # Calibration suggestion
            if not result.get('calibrated'):
                print("\n💡 Tip: Calibrate for personalized results")
                print("  Measure with a reference BP device, then run:")
                print(f"  python predict.py {video_path} --calibrate --sbp XXX --dbp YY")
            
            print("\nNote: This is an estimation tool, not medical advice.")
            print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
