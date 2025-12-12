"""CNN Training Pipeline for Blood Pressure Estimation.

This module provides a complete training pipeline for the 1D CNN model,
optimized for GPU training with mixed precision support.

Example:
    >>> from model_trainer_cnn import train_cnn_pipeline
    >>> metrics = train_cnn_pipeline(
    ...     data_path=['uci_dataset/Part_1.mat'],
    ...     epochs=100,
    ...     batch_size=32
    ... )
"""

from __future__ import annotations

import gc
import os
from typing import Optional, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import UCILoader
from model_cnn import (
    create_cnn_model,
    predict_with_cnn,
    save_cnn_model,
    train_cnn_model,
)


# =============================================================================
# Configuration Constants
# =============================================================================

SAMPLING_RATE = 125  # Hz
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS  # 625 samples


# =============================================================================
# Public API
# =============================================================================

def train_cnn_pipeline(
    data_path: Union[str, list[str]],
    output_path: str = "model/bp_model_cnn",
    max_windows: Optional[int] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42,
) -> dict:
    """Train a 1D CNN model for blood pressure estimation.
    
    This pipeline is optimized for GPU training with automatic mixed precision.
    
    Pipeline steps:
    1. Load raw PPG signals from UCI dataset
    2. Normalize signals (StandardScaler)
    3. Split into train/val/test sets
    4. Train CNN with early stopping
    5. Evaluate and save model

    Args:
        data_path: Path(s) to .mat file(s). Accepts single path or list.
        output_path: Base path for model files (without extension).
        max_windows: Limit windows per file (for debugging). None uses all.
        test_size: Fraction of data reserved for testing (0.0-1.0).
        val_size: Fraction of training data for validation.
        epochs: Maximum training epochs (early stopping may terminate earlier).
        batch_size: Batch size for training and inference.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing training, validation, and test metrics.

    Raises:
        FileNotFoundError: If data_path does not exist.
    """
    _print_header("CNN TRAINING PIPELINE")

    # Step 1: Load raw PPG data
    print("\n[Step 1/5] Loading raw PPG data...")
    X_raw, y = _load_data(data_path, max_windows)

    # Step 2: Normalize signals
    print("\n[Step 2/5] Normalizing PPG signals...")
    X_normalized, scaler = _normalize_signals(X_raw)
    
    # Free memory
    del X_raw
    gc.collect()

    # Step 3: Split data
    print("\n[Step 3/5] Splitting data...")
    splits = _split_data(X_normalized, y, test_size, val_size, random_state)
    X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']

    print(f"  Training samples:   {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples:       {len(X_test):,}")

    # Step 4: Train CNN model
    print("\n[Step 4/5] Training CNN model...")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Architecture: 4 Conv blocks + 2 Dense layers\n")

    model, history = train_cnn_model(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Step 5: Evaluate
    print("\n[Step 5/5] Evaluating model...")
    metrics = _evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
    _print_metrics(metrics)

    # Save model
    _save_model(model, scaler, metrics, history, batch_size, output_path)

    return metrics


# =============================================================================
# Private Helper Functions
# =============================================================================

def _load_data(
    data_path: Union[str, list[str]],
    max_windows: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load PPG signals and BP targets from UCI dataset."""
    data_paths = [data_path] if isinstance(data_path, str) else data_path
    print(f"Loading {len(data_paths)} file(s)...")

    loader = UCILoader(verbose=True)
    all_X_raw = []
    all_y = []

    for path in data_paths:
        print(f"\n  Processing {path}...")
        X_part, y_part = loader.load_part(path, max_windows=max_windows)
        all_X_raw.extend(X_part)
        all_y.append(y_part)

    # Convert to numpy arrays
    X_raw = np.array(all_X_raw)
    y = np.vstack(all_y) if len(all_y) > 1 else all_y[0]

    print(f"\n✓ Loaded {len(X_raw):,} windows")
    print(f"  PPG shape: {X_raw.shape}")
    print(f"  BP range - SBP: [{y[:, 0].min():.1f}, {y[:, 0].max():.1f}] mmHg")
    print(f"  BP range - DBP: [{y[:, 1].min():.1f}, {y[:, 1].max():.1f}] mmHg")

    # Clean up intermediate lists
    del all_X_raw, all_y
    gc.collect()

    return X_raw, y


def _normalize_signals(X_raw: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Normalize PPG signals to zero mean and unit variance."""
    print(f"  Input shape: {X_raw.shape}")
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_raw)
    
    print(f"  ✓ Normalized (mean=0, std=1)")
    
    return X_normalized, scaler


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int
) -> dict:
    """Split data into train/validation/test sets."""
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=random_state
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def _evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Compute metrics on train/val/test sets."""
    y_train_pred = predict_with_cnn(model, X_train)
    y_val_pred = predict_with_cnn(model, X_val)
    y_test_pred = predict_with_cnn(model, X_test)

    return {
        # Training metrics
        'train_sbp_mae': mean_absolute_error(y_train[:, 0], y_train_pred[:, 0]),
        'train_dbp_mae': mean_absolute_error(y_train[:, 1], y_train_pred[:, 1]),
        'train_overall_mae': mean_absolute_error(y_train, y_train_pred),
        
        # Validation metrics
        'val_sbp_mae': mean_absolute_error(y_val[:, 0], y_val_pred[:, 0]),
        'val_dbp_mae': mean_absolute_error(y_val[:, 1], y_val_pred[:, 1]),
        'val_overall_mae': mean_absolute_error(y_val, y_val_pred),
        
        # Test metrics
        'test_sbp_mae': mean_absolute_error(y_test[:, 0], y_test_pred[:, 0]),
        'test_dbp_mae': mean_absolute_error(y_test[:, 1], y_test_pred[:, 1]),
        'test_overall_mae': mean_absolute_error(y_test, y_test_pred),
        
        'test_sbp_rmse': np.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0])),
        'test_dbp_rmse': np.sqrt(mean_squared_error(y_test[:, 1], y_test_pred[:, 1])),
        
        'test_sbp_r2': r2_score(y_test[:, 0], y_test_pred[:, 0]),
        'test_dbp_r2': r2_score(y_test[:, 1], y_test_pred[:, 1]),
    }


def _save_model(
    model,
    scaler: StandardScaler,
    metrics: dict,
    history: dict,
    batch_size: int,
    output_path: str
) -> None:
    """Save trained model with metadata."""
    print(f"\n[Saving] Model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metadata = {
        'scaler': scaler,
        'metrics': metrics,
        'config': {
            'sampling_rate': SAMPLING_RATE,
            'window_seconds': WINDOW_SECONDS,
            'window_size': WINDOW_SIZE,
            'model_type': 'CNN',
            'epochs_trained': len(history['train_loss']),
            'batch_size': batch_size
        }
    }

    save_cnn_model(model, output_path, metadata)
    print("\n" + "=" * 70)


def _print_header(title: str) -> None:
    """Print a formatted section header."""
    print("=" * 70)
    print(title)
    print("=" * 70)


def _print_metrics(metrics: dict) -> None:
    """Print formatted training results."""
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


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train 1D CNN model for BP estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data', type=str, default='uci_dataset/Part_1.mat',
        help='Path to .mat file(s)'
    )
    parser.add_argument(
        '--output', type=str, default='model/bp_model_cnn',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--max-windows', type=int, default=None,
        help='Limit windows per file (for debugging)'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size'
    )

    args = parser.parse_args()
    train_cnn_pipeline(
        args.data,
        args.output,
        args.max_windows,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
