#!/usr/bin/env python3
"""Blood Pressure Estimation - Training Script

Trains the BP estimation model on UCI dataset parts.

Usage:
    python train.py                      # Train on all 4 parts (default)
    python train.py --parts 1 2          # Train on specific parts
    python train.py --max-windows 1000   # Limit windows per part (for testing)
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model_trainer import train_pipeline


def validate_dataset_parts(parts):
    """Validate that dataset files exist for the specified parts.
    
    Args:
        parts: List of part numbers
        
    Returns:
        List of valid file paths
        
    Raises:
        SystemExit: If any files are missing
    """
    base_path = Path("uci_dataset")
    missing_files = []
    valid_paths = []
    
    for part_num in parts:
        file_path = base_path / f"Part_{part_num}.mat"
        if file_path.exists():
            valid_paths.append(str(file_path))
        else:
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"✗ Error: Dataset file(s) not found:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease ensure UCI dataset files are in uci_dataset/")
        sys.exit(1)
    
    return valid_paths


def print_training_summary(parts, max_windows, output_path, test_size):
    """Print training configuration summary."""
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Dataset Parts:  {', '.join([f'Part_{p}' for p in parts])}")
    print(f"Training Mode:  {'DEBUG' if max_windows else 'FULL'}")
    if max_windows:
        print(f"Max Windows:    {max_windows} per part")
    print(f"Output Model:   {output_path}")
    print(f"Test Split:     {test_size:.0%}")
    print("=" * 70)
    print()


def evaluate_results(metrics):
    """Evaluate and print training results assessment.
    
    Args:
        metrics: Dictionary of training metrics
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 70)
    
    sbp_mae = metrics['test_sbp_mae']
    dbp_mae = metrics['test_dbp_mae']
    
    # Performance thresholds
    EXCELLENT_SBP = 5.0
    EXCELLENT_DBP = 3.0
    ACCEPTABLE_SBP = 15.0
    ACCEPTABLE_DBP = 12.0
    
    # SBP assessment
    if sbp_mae < EXCELLENT_SBP:
        sbp_status = "✓ EXCELLENT"
    elif sbp_mae < ACCEPTABLE_SBP:
        sbp_status = "✓ Good"
    else:
        sbp_status = "⚠ Needs improvement"
    
    # DBP assessment
    if dbp_mae < EXCELLENT_DBP:
        dbp_status = "✓ EXCELLENT"
    elif dbp_mae < ACCEPTABLE_DBP:
        dbp_status = "✓ Good"
    else:
        dbp_status = "⚠ Needs improvement"
    
    print(f"SBP MAE: {sbp_mae:.2f} mmHg  {sbp_status}")
    print(f"DBP MAE: {dbp_mae:.2f} mmHg  {dbp_status}")
    
    # Overall recommendation
    if sbp_mae < ACCEPTABLE_SBP and dbp_mae < ACCEPTABLE_DBP:
        print("\n✓ Model meets clinical accuracy targets")
        if sbp_mae < EXCELLENT_SBP and dbp_mae < EXCELLENT_DBP:
            print("  Performance is excellent - ready for deployment")
    else:
        print("\n⚠ Model needs improvement:")
        if sbp_mae >= ACCEPTABLE_SBP:
            print(f"  - SBP MAE should be < {ACCEPTABLE_SBP} mmHg")
        if dbp_mae >= ACCEPTABLE_DBP:
            print(f"  - DBP MAE should be < {ACCEPTABLE_DBP} mmHg")
        print("  Consider training on more data parts")
    
    print("=" * 70)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train Blood Pressure Estimation Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training on all parts (recommended)
  python train.py
  
  # Train on specific parts only
  python train.py --parts 1 2
  
  # Quick test with limited data
  python train.py --parts 1 --max-windows 500
  
  # Custom output location
  python train.py --output models/my_custom_model.pkl
        """
    )
    
    parser.add_argument(
        '--parts',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4],
        metavar='N',
        help='Dataset parts to use (default: all 4)'
    )
    
    parser.add_argument(
        '--max-windows',
        type=int,
        default=None,
        metavar='N',
        help='Max windows per part (for testing, default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='model/bp_model.pkl',
        metavar='PATH',
        help='Output model path (default: model/bp_model.pkl)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        metavar='FRAC',
        help='Test set fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['rf', 'cnn'],
        default='rf',
        help='Model type: rf (RandomForest) or cnn (CNN, default: rf)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='CNN epochs (default: 100, only for --model cnn)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        metavar='N',
        help='CNN batch size (default: 32, only for --model cnn)'
    )
    
    args = parser.parse_args()
    
    # Validate dataset files exist
    data_paths = validate_dataset_parts(args.parts)
    
    # Print configuration
    print_training_summary(args.parts, args.max_windows, args.output, args.test_size)
    
    # Add model type to summary
    model_name = "Random Forest" if args.model == 'rf' else "1D CNN"
    print(f"Model Type:     {model_name}")
    if args.model == 'cnn':
        print(f"CNN Epochs:     {args.epochs}")
        print(f"Batch Size:     {args.batch_size}")
    print("=" * 70)
    print()
    
    try:
        # Select appropriate training pipeline
        if args.model == 'cnn':
            from src.model_trainer_cnn import train_cnn_pipeline
            metrics = train_cnn_pipeline(
                data_path=data_paths,
                output_path=args.output.replace('.pkl', '_cnn'),
                max_windows=args.max_windows,
                test_size=args.test_size,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        else:
            # Random Forest (existing)
            metrics = train_pipeline(
                data_path=data_paths,
                output_path=args.output,
                max_windows=args.max_windows,
                test_size=args.test_size
            )
        
        # Evaluate results
        evaluate_results(metrics)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        
        # Show traceback in debug mode
        if os.getenv('DEBUG'):
            import traceback
            traceback.print_exc()
        else:
            print("\nSet DEBUG=1 for full traceback")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
