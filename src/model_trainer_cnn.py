"""CNN Training Pipeline for Blood Pressure Estimation.

This module provides a training pipeline specifically for CNN models
that work with raw PPG signals.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import gc
from typing import Union, List

from data_loader import UCILoader
from model_cnn import create_cnn_model, train_cnn_model, save_cnn_model, predict_with_cnn


def train_cnn_pipeline(
    data_path: Union[str, List[str]],
    output_path: str = "model/bp_model_cnn",
    max_windows: int = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42
) -> dict:
    """Complete CNN training pipeline for BP estimation.
    
    Args:
        data_path: Path(s) to .mat file(s)
        output_path: Where to save the trained model (without extension)
        max_windows: Maximum windows to use per file (None = use all)
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        epochs: Maximum training epochs
        batch_size: Batch size for CNN training
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with training metrics
    """
    print("=" * 70)
    print("CNN TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load data (raw PPG signals)
    print("\n[Step 1/5] Loading raw PPG data...")
    
    # Convert single path to list
    if isinstance(data_path, str):
        data_paths = [data_path]
    else:
        data_paths = data_path
    
    print(f"Loading {data_paths}...")
    
    loader = UCILoader(verbose=True)
    
    # Load data from all files
    all_X_raw = []
    all_y = []
    
    for path in data_paths:
        print(f"\n  Processing {path}...")
        X_raw_part, y_part = loader.load_part(path, max_windows=max_windows)
        all_X_raw.extend(X_raw_part)
        all_y.append(y_part)
    
    # Combine all data
    print(f"\n  Combining data from {len(data_paths)} parts...")
    print(f"  Total windows collected: {len(all_X_raw)}")
    print(f"  Converting to numpy array...")
    import sys
    sys.stdout.flush()
    
    X_raw = np.array(all_X_raw)  # Convert to numpy array (N, 625)
    print(f"  ✓ PPG array created: {X_raw.shape}")
    sys.stdout.flush()
    
    print(f"  Stacking BP targets...")
    sys.stdout.flush()
    y = np.vstack(all_y) if len(all_y) > 1 else all_y[0]
    print(f"  ✓ BP targets array: {y.shape}")
    sys.stdout.flush()
    
    print(f"\nLoaded {len(X_raw)} windows with BP targets")
    print(f"PPG signal shape: {X_raw.shape}")
    print(f"BP range - SBP: [{y[:, 0].min():.1f}, {y[:, 0].max():.1f}] mmHg")
    print(f"BP range - DBP: [{y[:, 1].min():.1f}, {y[:, 1].max():.1f}] mmHg")
    
    # Clean up raw lists to free memory
    del all_X_raw
    del all_y
    gc.collect()
    
    # Step 2: Normalize PPG signals
    print("\n[Step 2/5] Normalizing PPG signals...")
    print(f"  Input shape: {X_raw.shape}")
    import sys
    sys.stdout.flush()
    
    scaler = StandardScaler()
    print(f"  Fitting scaler...")
    sys.stdout.flush()
    X_normalized = scaler.fit_transform(X_raw)
    print(f"  ✓ Signal normalized (mean=0, std=1)")
    print(f"  Output shape: {X_normalized.shape}")
    sys.stdout.flush()
    
    # Clean up raw array
    del X_raw
    gc.collect()
    
    # Step 3: Split data (train/val/test)
    print("\n[Step 3/5] Splitting data...")
    sys.stdout.flush()
    
    import time
    start_time = time.time()
    
    # First split: train+val vs test
    print(f"  First split: Separating test set ({test_size:.0%} of {len(X_normalized)} samples)...")
    sys.stdout.flush()
    split_start = time.time()
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_normalized, y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"  ✓ First split complete ({time.time() - split_start:.1f}s)")
    print(f"    Train+Val: {len(X_trainval)}, Test: {len(X_test)}")
    sys.stdout.flush()
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    print(f"  Second split: Separating validation set ({val_size_adjusted:.0%} of {len(X_trainval)} samples)...")
    sys.stdout.flush()
    split_start = time.time()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    print(f"  ✓ Second split complete ({time.time() - split_start:.1f}s)")
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  ✓ Total splitting time: {time.time() - start_time:.1f}s")
    sys.stdout.flush()
    
    print(f"  Training samples:    {len(X_train)}")
    print(f"  Validation samples:  {len(X_val)}")
    print(f"  Test samples:        {len(X_test)}")
    
    # Step 4: Train CNN model
    print("\n[Step 4/5] Training CNN model...")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Architecture: 4 Conv blocks + 2 Dense layers")
    print()
    
    model, history = train_cnn_model(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Step 5: Evaluate
    print("\n[Step 5/5] Evaluating model...")
    
    # Predictions using the correct function
    y_train_pred = predict_with_cnn(model, X_train)
    y_val_pred = predict_with_cnn(model, X_val)
    y_test_pred = predict_with_cnn(model, X_test)
    
    # Calculate metrics
    metrics = {
        'train_sbp_mae': mean_absolute_error(y_train[:, 0], y_train_pred[:, 0]),
        'train_dbp_mae': mean_absolute_error(y_train[:, 1], y_train_pred[:, 1]),
        'train_overall_mae': mean_absolute_error(y_train, y_train_pred),
        
        'val_sbp_mae': mean_absolute_error(y_val[:, 0], y_val_pred[:, 0]),
        'val_dbp_mae': mean_absolute_error(y_val[:, 1], y_val_pred[:, 1]),
        'val_overall_mae': mean_absolute_error(y_val, y_val_pred),
        
        'test_sbp_mae': mean_absolute_error(y_test[:, 0], y_test_pred[:, 0]),
        'test_dbp_mae': mean_absolute_error(y_test[:, 1], y_test_pred[:, 1]),
        'test_overall_mae': mean_absolute_error(y_test, y_test_pred),
        
        'test_sbp_rmse': np.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0])),
        'test_dbp_rmse': np.sqrt(mean_squared_error(y_test[:, 1], y_test_pred[:, 1])),
        
        'test_sbp_r2': r2_score(y_test[:, 0], y_test_pred[:, 0]),
        'test_dbp_r2': r2_score(y_test[:, 1], y_test_pred[:, 1]),
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("CNN TRAINING RESULTS")
    print("=" * 70)
    
    print("\nTraining Set:")
    print(f"  SBP MAE: {metrics['train_sbp_mae']:.2f} mmHg")
    print(f"  DBP MAE: {metrics['train_dbp_mae']:.2f} mmHg")
    print(f"  Overall MAE: {metrics['train_overall_mae']:.2f} mmHg")
    
    print("\nValidation Set:")
    print(f"  SBP MAE: {metrics['val_sbp_mae']:.2f} mmHg")
    print(f"  DBP MAE: {metrics['val_dbp_mae']:.2f} mmHg")
    print(f"  Overall MAE: {metrics['val_overall_mae']:.2f} mmHg")
    
    print("\nTest Set:")
    print(f"  SBP MAE: {metrics['test_sbp_mae']:.2f} mmHg")
    print(f"  DBP MAE: {metrics['test_dbp_mae']:.2f} mmHg")
    print(f"  Overall MAE: {metrics['test_overall_mae']:.2f} mmHg")
    print(f"  R² Score (SBP): {metrics['test_sbp_r2']:.3f}")
    print(f"  R² Score (DBP): {metrics['test_dbp_r2']:.3f}")
    
    # Save model
    print(f"\n[Saving] Model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    metadata = {
        'scaler': scaler,
        'metrics': metrics,
        'config': {
            'sampling_rate': 125,
            'window_seconds': 5,
            'window_size': 625,
            'model_type': 'CNN',
            'epochs_trained': len(history['train_loss']),
            'batch_size': batch_size
        }
    }
    
    save_cnn_model(model, output_path, metadata)
    
    print("\n" + "=" * 70)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN BP estimation model")
    parser.add_argument('--data', type=str, default='uci_dataset/Part_1.mat',
                        help='Path to .mat file')
    parser.add_argument('--output', type=str, default='model/bp_model_cnn',
                        help='Output path for trained model')
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Max windows to use (for testing)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    train_cnn_pipeline(args.data, args.output, args.max_windows,
                      epochs=args.epochs, batch_size=args.batch_size)
