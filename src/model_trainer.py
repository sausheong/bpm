"""Model Training Pipeline for Blood Pressure Estimation.

This module orchestrates the complete training workflow:
1. Load data using UCILoader
2. Extract features using PPGFeatureExtractor
3. Train RandomForestRegressor
4. Evaluate and save model with feature names
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Union, List

from data_loader import UCILoader
from features import PPGFeatureExtractor


def train_pipeline(
    data_path: Union[str, List[str]],
    output_path: str = "model/bp_model.pkl",
    max_windows: int = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """Complete training pipeline for BP estimation.
    
    Args:
        data_path: Path(s) to .mat file(s). Can be:
            - Single path: 'uci_dataset/Part_1.mat'
            - Multiple paths: ['uci_dataset/Part_1.mat', 'uci_dataset/Part_2.mat']
        output_path: Where to save the trained model
        max_windows: Maximum windows to use per file (None = use all)
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with training metrics
    """
    print("=" * 70)
    print("BLOOD PRESSURE ESTIMATION - TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1/5] Loading data...")
    
    # Convert single path to list for uniform handling
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
    X_raw = all_X_raw
    y = np.vstack(all_y) if len(all_y) > 1 else all_y[0]
    
    print(f"\nLoaded {len(X_raw)} windows with BP targets")
    print(f"BP range - SBP: [{y[:, 0].min():.1f}, {y[:, 0].max():.1f}] mmHg")
    print(f"BP range - DBP: [{y[:, 1].min():.1f}, {y[:, 1].max():.1f}] mmHg")
    
    # Step 2: Extract features
    print("\n[Step 2/5] Extracting features...")
    extractor = PPGFeatureExtractor(sampling_rate=125, verbose=True)
    X_features, valid_indices = extractor.transform(X_raw)
    
    # Align targets with successfully extracted features
    y_aligned = y[valid_indices]
    
    print(f"\nFeature extraction complete:")
    print(f"  Valid samples: {len(valid_indices)} / {len(X_raw)}")
    print(f"  Features per sample: {X_features.shape[1]}")
    
    # Step 3: Train/test split
    print("\n[Step 3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_aligned,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Step 4: Train model
    print("\n[Step 4/5] Training RandomForest model...")
    
    # Optional: Standardize features (improves some models, though RF is robust)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configure RandomForest for multi-output regression
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Step 5: Evaluate
    print("\n[Step 5/5] Evaluating model...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = evaluate_model(y_train, y_train_pred, y_test, y_test_pred)
    
    # Print results
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
    
    # Save model
    print(f"\n[Saving] Model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X_features.columns),
        'metrics': metrics,
        'config': {
            'sampling_rate': 125,
            'window_seconds': 5,
            'n_features': X_features.shape[1]
        }
    }
    
    joblib.dump(model_package, output_path)
    print(f"Model saved successfully!")
    
    print("\n" + "=" * 70)
    
    return metrics


def evaluate_model(y_train_true, y_train_pred, y_test_true, y_test_pred) -> dict:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        y_train_true: True training labels (N, 2)
        y_train_pred: Predicted training labels (N, 2)
        y_test_true: True test labels (N, 2)
        y_test_pred: Predicted test labels (N, 2)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Training metrics
    metrics['train_sbp_mae'] = mean_absolute_error(y_train_true[:, 0], y_train_pred[:, 0])
    metrics['train_dbp_mae'] = mean_absolute_error(y_train_true[:, 1], y_train_pred[:, 1])
    metrics['train_overall_mae'] = mean_absolute_error(y_train_true, y_train_pred)
    
    # Test metrics
    metrics['test_sbp_mae'] = mean_absolute_error(y_test_true[:, 0], y_test_pred[:, 0])
    metrics['test_dbp_mae'] = mean_absolute_error(y_test_true[:, 1], y_test_pred[:, 1])
    metrics['test_overall_mae'] = mean_absolute_error(y_test_true, y_test_pred)
    
    metrics['test_sbp_rmse'] = np.sqrt(mean_squared_error(y_test_true[:, 0], y_test_pred[:, 0]))
    metrics['test_dbp_rmse'] = np.sqrt(mean_squared_error(y_test_true[:, 1], y_test_pred[:, 1]))
    
    metrics['test_sbp_r2'] = r2_score(y_test_true[:, 0], y_test_pred[:, 0])
    metrics['test_dbp_r2'] = r2_score(y_test_true[:, 1], y_test_pred[:, 1])
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BP estimation model")
    parser.add_argument('--data', type=str, default='uci_dataset/Part_1.mat',
                        help='Path to .mat file')
    parser.add_argument('--output', type=str, default='model/bp_model.pkl',
                        help='Output path for trained model')
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Max windows to use (for testing)')
    
    args = parser.parse_args()
    
    train_pipeline(args.data, args.output, args.max_windows)
