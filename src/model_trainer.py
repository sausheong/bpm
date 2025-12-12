"""Random Forest Training Pipeline for Blood Pressure Estimation.

This module provides a complete training pipeline for the Random Forest model,
including data loading, feature extraction, model training, and evaluation.

Example:
    >>> from model_trainer import train_pipeline
    >>> metrics = train_pipeline(
    ...     data_path=['uci_dataset/Part_1.mat', 'uci_dataset/Part_2.mat'],
    ...     output_path='model/bp_model.pkl',
    ...     test_size=0.2
    ... )
"""

from __future__ import annotations

import os
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import UCILoader
from features import PPGFeatureExtractor


# =============================================================================
# Configuration Constants
# =============================================================================

SAMPLING_RATE = 125  # Hz
WINDOW_SECONDS = 5
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS  # 625 samples

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2


# =============================================================================
# Public API
# =============================================================================

def train_pipeline(
    data_path: Union[str, list[str]],
    output_path: str = "model/bp_model.pkl",
    max_windows: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a Random Forest model for blood pressure estimation.
    
    This pipeline performs the following steps:
    1. Load PPG and ABP signals from UCI dataset
    2. Extract morphological features using NeuroKit2
    3. Train a multi-output Random Forest regressor
    4. Evaluate and save the model

    Args:
        data_path: Path(s) to .mat file(s). Accepts single path or list.
        output_path: Where to save the trained model package.
        max_windows: Limit windows per file (for debugging). None uses all.
        test_size: Fraction of data reserved for testing (0.0-1.0).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing training and test metrics:
            - train_sbp_mae, train_dbp_mae, train_overall_mae
            - test_sbp_mae, test_dbp_mae, test_overall_mae
            - test_sbp_r2, test_dbp_r2

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If no valid windows are extracted.
    """
    _print_header("RANDOM FOREST TRAINING PIPELINE")

    # Step 1: Load data
    print("\n[Step 1/5] Loading data...")
    X_raw, y = _load_data(data_path, max_windows)

    # Step 2: Extract features
    print("\n[Step 2/5] Extracting features...")
    X_features, valid_indices = _extract_features(X_raw)
    y_aligned = y[valid_indices]

    if len(valid_indices) == 0:
        raise ValueError("No valid windows after feature extraction")

    # Step 3: Split data
    print("\n[Step 3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_aligned,
        test_size=test_size,
        random_state=random_state
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")

    # Step 4: Train model
    print("\n[Step 4/5] Training RandomForest model...")
    model, scaler = _train_random_forest(X_train, y_train, random_state)

    # Step 5: Evaluate
    print("\n[Step 5/5] Evaluating model...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    metrics = compute_metrics(y_train, y_train_pred, y_test, y_test_pred)
    _print_metrics(metrics)

    # Save model package
    _save_model_package(
        model=model,
        scaler=scaler,
        feature_names=list(X_features.columns),
        metrics=metrics,
        output_path=output_path
    )

    return metrics


def compute_metrics(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray
) -> dict:
    """Compute comprehensive evaluation metrics for BP prediction.
    
    Args:
        y_train_true: Ground truth training labels (N, 2) [SBP, DBP].
        y_train_pred: Predicted training labels (N, 2).
        y_test_true: Ground truth test labels (N, 2).
        y_test_pred: Predicted test labels (N, 2).

    Returns:
        Dictionary with MAE, RMSE, and R² scores for SBP and DBP.
    """
    return {
        # Training metrics
        'train_sbp_mae': mean_absolute_error(y_train_true[:, 0], y_train_pred[:, 0]),
        'train_dbp_mae': mean_absolute_error(y_train_true[:, 1], y_train_pred[:, 1]),
        'train_overall_mae': mean_absolute_error(y_train_true, y_train_pred),
        
        # Test metrics
        'test_sbp_mae': mean_absolute_error(y_test_true[:, 0], y_test_pred[:, 0]),
        'test_dbp_mae': mean_absolute_error(y_test_true[:, 1], y_test_pred[:, 1]),
        'test_overall_mae': mean_absolute_error(y_test_true, y_test_pred),
        
        'test_sbp_rmse': np.sqrt(mean_squared_error(y_test_true[:, 0], y_test_pred[:, 0])),
        'test_dbp_rmse': np.sqrt(mean_squared_error(y_test_true[:, 1], y_test_pred[:, 1])),
        
        'test_sbp_r2': r2_score(y_test_true[:, 0], y_test_pred[:, 0]),
        'test_dbp_r2': r2_score(y_test_true[:, 1], y_test_pred[:, 1]),
    }


# =============================================================================
# Private Helper Functions
# =============================================================================

def _load_data(
    data_path: Union[str, list[str]],
    max_windows: Optional[int] = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load PPG signals and BP targets from UCI dataset.
    
    Handles both single file and multiple file loading.
    """
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

    y = np.vstack(all_y) if len(all_y) > 1 else all_y[0]

    print(f"\n✓ Loaded {len(all_X_raw)} windows")
    print(f"  BP range - SBP: [{y[:, 0].min():.1f}, {y[:, 0].max():.1f}] mmHg")
    print(f"  BP range - DBP: [{y[:, 1].min():.1f}, {y[:, 1].max():.1f}] mmHg")

    return all_X_raw, y


def _extract_features(X_raw: list[np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract morphological features from raw PPG signals."""
    extractor = PPGFeatureExtractor(sampling_rate=SAMPLING_RATE, verbose=True)
    X_features, valid_indices = extractor.transform(X_raw)

    print(f"\n✓ Feature extraction complete")
    print(f"  Valid samples: {len(valid_indices)} / {len(X_raw)}")
    print(f"  Features: {X_features.shape[1]}")

    return X_features, valid_indices


def _train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int
) -> tuple[RandomForestRegressor, StandardScaler]:
    """Train a Random Forest regressor with standardized features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def _save_model_package(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    feature_names: list[str],
    metrics: dict,
    output_path: str
) -> None:
    """Save model, scaler, and metadata as a single package."""
    print(f"\n[Saving] Model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'config': {
            'sampling_rate': SAMPLING_RATE,
            'window_seconds': WINDOW_SECONDS,
            'n_features': len(feature_names)
        }
    }

    joblib.dump(package, output_path)
    print(f"✓ Model saved successfully!")


def _print_header(title: str) -> None:
    """Print a formatted section header."""
    print("=" * 70)
    print(title)
    print("=" * 70)


def _print_metrics(metrics: dict) -> None:
    """Print formatted training results."""
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    print("\nTraining Set:")
    print(f"  SBP MAE: {metrics['train_sbp_mae']:.2f} mmHg")
    print(f"  DBP MAE: {metrics['train_dbp_mae']:.2f} mmHg")
    print(f"  Overall MAE: {metrics['train_overall_mae']:.2f} mmHg")

    print("\nTest Set:")
    print(f"  SBP MAE: {metrics['test_sbp_mae']:.2f} mmHg")
    print(f"  DBP MAE: {metrics['test_dbp_mae']:.2f} mmHg")
    print(f"  Overall MAE: {metrics['test_overall_mae']:.2f} mmHg")
    print(f"  R² Score (SBP): {metrics['test_sbp_r2']:.3f}")
    print(f"  R² Score (DBP): {metrics['test_dbp_r2']:.3f}")

    print("\n" + "=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Random Forest model for BP estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data', type=str, default='uci_dataset/Part_1.mat',
        help='Path to .mat file(s)'
    )
    parser.add_argument(
        '--output', type=str, default='model/bp_model.pkl',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--max-windows', type=int, default=None,
        help='Limit windows per file (for debugging)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction of data for testing'
    )

    args = parser.parse_args()
    train_pipeline(args.data, args.output, args.max_windows, args.test_size)
