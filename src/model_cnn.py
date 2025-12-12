"""1D CNN Model for Blood Pressure Estimation.

This module implements a 1D Convolutional Neural Network that learns
features directly from raw PPG signals, avoiding manual feature engineering.

Architecture:
    - 4 convolutional blocks with batch normalization and max pooling
    - Global average pooling
    - 2 fully connected layers with dropout
    - Multi-output regression (SBP, DBP)

Example:
    >>> from model_cnn import create_cnn_model, train_cnn_model
    >>> model = create_cnn_model()
    >>> model, history = train_cnn_model(X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# Suppress sklearn version warnings (e.g. 1.7.0 vs 1.8.0 for StandardScaler)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# =============================================================================
# Configuration Constants
# =============================================================================

WINDOW_SIZE = 625  # 5 seconds @ 125 Hz
OUTPUT_SIZE = 2    # SBP, DBP

# Training defaults
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 7
DROPOUT_RATE = 0.3


# =============================================================================
# Dataset
# =============================================================================

class PPGDataset(Dataset):
    """PyTorch Dataset for PPG signals with lazy tensor conversion.
    
    Stores data as numpy arrays and converts to tensors only when accessing
    items, reducing peak memory usage for large datasets.
    
    Args:
        X: PPG signals of shape (N, 625).
        y: BP targets of shape (N, 2) containing [SBP, DBP].
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_tensor = torch.from_numpy(self.X[idx]).float().unsqueeze(0)
        y_tensor = torch.from_numpy(self.y[idx]).float()
        return x_tensor, y_tensor


# =============================================================================
# Model Architecture
# =============================================================================

class CNNModel(nn.Module):
    """1D Convolutional Neural Network for BP estimation.
    
    Architecture:
        Input: (batch, 1, 625)
        → Conv1D(32, k=7) + BN + ReLU + MaxPool
        → Conv1D(64, k=5) + BN + ReLU + MaxPool
        → Conv1D(128, k=3) + BN + ReLU + MaxPool
        → Conv1D(128, k=3) + BN + ReLU
        → Global Average Pooling
        → FC(128) + ReLU + Dropout
        → FC(64) + ReLU + Dropout
        → FC(2) → [SBP, DBP]
    
    Total Parameters: ~110,146
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            # Block 1: Low-level features
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 2: Mid-level features
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 3: High-level features
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 4: Abstract features
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(64, OUTPUT_SIZE),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, 625).
            
        Returns:
            Output tensor of shape (batch, 2) with [SBP, DBP] predictions.
        """
        x = self.conv_blocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x


# =============================================================================
# Public API
# =============================================================================

def create_cnn_model() -> CNNModel:
    """Create a new CNN model instance.
    
    Returns:
        Uninitialized CNNModel ready for training.
    """
    return CNNModel()


def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: Optional[str] = None,
    verbose: int = 1,
) -> tuple[CNNModel, dict]:
    """Train CNN model with early stopping and mixed precision.
    
    Features:
        - Automatic device detection (CUDA > MPS > CPU)
        - Mixed precision training on CUDA for faster training
        - Early stopping with patience
        - Learning rate reduction on plateau

    Args:
        X_train: Training PPG signals of shape (N, 625).
        y_train: Training BP targets of shape (N, 2).
        X_val: Validation PPG signals.
        y_val: Validation BP targets.
        epochs: Maximum training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate for Adam optimizer.
        device: Device to use ('cpu', 'cuda', 'mps'). Auto-detects if None.
        verbose: Verbosity level (0=silent, 1=progress every 10 epochs).

    Returns:
        Tuple of (trained_model, training_history).
        History contains 'train_loss' and 'val_loss' lists.
    """
    device = _detect_device(device)
    
    # Create data loaders
    train_loader = DataLoader(
        PPGDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        PPGDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )
    
    if verbose:
        print(f"  ✓ DataLoaders: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Initialize model and training components
    model = create_cnn_model().to(device)
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE
    )
    
    # Mixed precision for CUDA
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    if scaler and verbose:
        print(f"  ✓ Mixed Precision (AMP) enabled")
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validation phase
        val_loss = _validate_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    
    return model, history


def save_cnn_model(
    model: CNNModel,
    output_path: str,
    metadata: Optional[dict] = None
) -> None:
    """Save CNN model weights and metadata.
    
    Creates two files:
        - {output_path}.pt: Model weights
        - {output_path}_metadata.pkl: Scaler, metrics, config

    Args:
        model: Trained PyTorch model.
        output_path: Base path for saving (without extension).
        metadata: Optional dictionary with scaler, metrics, etc.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    torch.save(model.state_dict(), f"{output_path}.pt")
    
    if metadata:
        joblib.dump(metadata, f"{output_path}_metadata.pkl")
    
    print(f"✓ Model saved:")
    print(f"  - {output_path}.pt")
    if metadata:
        print(f"  - {output_path}_metadata.pkl")


def load_cnn_model(model_path: str) -> tuple[CNNModel, dict]:
    """Load a saved CNN model and metadata.
    
    Args:
        model_path: Base path to model files (without extension).

    Returns:
        Tuple of (model, metadata). Model is in eval mode.
    """
    model = create_cnn_model()
    # map_location='cpu' ensures we can load CUDA-saved models on CPU/MPS
    # The model will be moved to the correct device later in predict_with_cnn
    model.load_state_dict(torch.load(f"{model_path}.pt", map_location='cpu', weights_only=True))
    model.eval()
    
    metadata_path = f"{model_path}_metadata.pkl"
    metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
    
    return model, metadata


def predict_with_cnn(
    model: CNNModel,
    ppg_signals: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Make predictions using trained CNN model.
    
    Processes data in batches to avoid out-of-memory errors on large datasets.

    Args:
        model: Trained CNN model.
        ppg_signals: PPG signals of shape (N, 625) or (625,).
        device: Device to use. Auto-detects if None.
        batch_size: Batch size for inference.

    Returns:
        Predictions array of shape (N, 2) with [SBP, DBP].
    """
    device = _detect_device(device, verbose=False)
    model.to(device)
    model.eval()
    
    # Handle single signal
    if ppg_signals.ndim == 1:
        ppg_signals = ppg_signals.reshape(1, -1)
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(ppg_signals), batch_size):
            batch = torch.FloatTensor(ppg_signals[i:i+batch_size]).unsqueeze(1).to(device)
            
            if device == 'cuda':
                with torch.amp.autocast('cuda'):
                    preds = model(batch).float().cpu().numpy()
            else:
                preds = model(batch).cpu().numpy()
            
            predictions.append(preds)
    
    return np.vstack(predictions)


# =============================================================================
# Private Helper Functions
# =============================================================================

def _detect_device(device: Optional[str], verbose: bool = True) -> str:
    """Detect the best available device for training."""
    if device is not None:
        return device
    
    if torch.backends.mps.is_available():
        if verbose:
            print("🚀 Using Apple Silicon GPU (MPS)")
        return 'mps'
    elif torch.cuda.is_available():
        if verbose:
            print("🚀 Using NVIDIA GPU (CUDA)")
        return 'cuda'
    else:
        if verbose:
            print("ℹ️  Using CPU (no GPU available)")
        return 'cpu'


def _train_epoch(
    model: CNNModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: str,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
        total_samples += len(X_batch)
    
    return total_loss / total_samples


def _validate_epoch(
    model: CNNModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Run one validation epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)
    
    return total_loss / total_samples


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Creating CNN model...")
    model = create_cnn_model()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Expected input: (batch, 1, {WINDOW_SIZE})")
    print(f"Output: (batch, {OUTPUT_SIZE}) [SBP, DBP]")
    
    # Test forward pass
    test_input = torch.randn(4, 1, WINDOW_SIZE)
    test_output = model(test_input)
    print(f"\nTest forward pass:")
    print(f"  Input: {tuple(test_input.shape)}")
    print(f"  Output: {tuple(test_output.shape)}")
