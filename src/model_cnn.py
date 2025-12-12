"""CNN Model Architecture for Blood Pressure Estimation (PyTorch).

This module implements a 1D Convolutional Neural Network that learns
features directly from raw PPG signals, avoiding manual feature engineering.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import os


class PPGDataset(Dataset):
    """PyTorch Dataset for PPG signals."""
    
    def __init__(self, X, y):
        """Initialize dataset.
        
        Args:
            X: PPG signals (N, 625)
            y: BP targets (N,  2)
        """
        self.X = X  # Keep as numpy array
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Lazy conversion to tensor
        x_tensor = torch.from_numpy(self.X[idx]).float().unsqueeze(0)  # (1, 625)
        y_tensor = torch.from_numpy(self.y[idx]).float()
        return x_tensor, y_tensor


class CNNModel(nn.Module):
    """1D CNN for BP estimation from raw PPG signals.
    
    Architecture:
        - 4 convolutional blocks with batch norm and pooling
        - Global average pooling
        - 2 dense layers with dropout
        - Multi-output regression (SBP, DBP)
    """
    
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Conv Block 1: Extract low-level features
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv Block 2: Mid-level features
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Conv Block 3: High-level features
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Conv Block 4: Abstract features
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output layer (2 values: SBP, DBP)
        self.fc3 = nn.Linear(64, 2)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.gap(x).squeeze(-1)
        
        # Dense layers
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def create_cnn_model():
    """Create CNN model instance."""
    return CNNModel()


def train_cnn_model(X_train, y_train, X_val, y_val,
                    epochs=100, batch_size=32, learning_rate=0.001,
                    device=None, verbose=1):
    """Train CNN model with early stopping.
    
    Args:
        X_train: Training PPG signals (N, 625)
        y_train: Training BP targets (N, 2)
        X_val: Validation PPG signals
        y_val: Validation BP targets
        epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
        verbose: Verbosity level
    
    Returns:
        Trained model and training history
    """
    # Auto-detect best available device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
            print(f"🚀 Using Apple Silicon GPU (MPS) for training")
        elif torch.cuda.is_available():
            device = 'cuda'
            print(f"🚀 Using NVIDIA GPU (CUDA) for training")
        else:
            device = 'cpu'
            print(f"ℹ️  Using CPU for training (no GPU available)")
    
    print(f"   Device: {device}")
    
    import time
    import sys
    init_start = time.time()
    
    # Create datasets
    print(f"  Creating PyTorch datasets...")
    sys.stdout.flush()
    dataset_start = time.time()
    
    train_dataset = PPGDataset(X_train, y_train)
    val_dataset = PPGDataset(X_val, y_val)
    
    print(f"  ✓ Datasets created ({time.time() - dataset_start:.1f}s)")
    sys.stdout.flush()
    
    print(f"  Creating DataLoaders (batch_size={batch_size})...")
    sys.stdout.flush()
    loader_start = time.time()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  ✓ DataLoaders created ({time.time() - loader_start:.1f}s)")
    print(f"    Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    sys.stdout.flush()
    
    # Create model
    print(f"  Initializing CNN model on {device}...")
    sys.stdout.flush()
    model_start = time.time()
    
    model = create_cnn_model().to(device)
    
    print(f"  ✓ Model on {device} ({time.time() - model_start:.1f}s)")
    sys.stdout.flush()
    
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    
    # Initialize Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    if scaler:
        print(f"  ✓ Mixed Precision (AMP) enabled")
    
    print(f"  ✓ Total initialization: {time.time() - init_start:.1f}s")
    print()
    sys.stdout.flush()
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard Training (CPU/MPS)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)
        
        val_loss /= len(val_dataset)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Stop if no improvement
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    return model, history


def save_cnn_model(model, output_path, metadata=None):
    """Save CNN model and metadata.
    
    Args:
        model: Trained PyTorch model
        output_path: Base path for saving
        metadata: Optional metadata dictionary
    """
    # Save model state
    torch.save(model.state_dict(), f"{output_path}.pt")
    
    # Save metadata
    if metadata:
        joblib.dump(metadata, f"{output_path}_metadata.pkl")
    
    print(f"✓ Model saved:")
    print(f"  - {output_path}.pt")
    if metadata:
        print(f"  - {output_path}_metadata.pkl")


def load_cnn_model(model_path):
    """Load saved CNN model.
    
    Args:
        model_path: Path to model file (without extension)
    
    Returns:
        Loaded model and metadata
    """
    model = create_cnn_model()
    model.load_state_dict(torch.load(f"{model_path}.pt", weights_only=True))
    model.eval()
    
    # Load metadata
    metadata_path = f"{model_path}_metadata.pkl"
    metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
    
    return model, metadata


def predict_with_cnn(model, ppg_signals, device=None, batch_size=32):
    """Make predictions using CNN model.
    
    Args:
        model: Trained PyTorch model
        ppg_signals: PPG signals (N, 625) or (625,)
        device: Device to use (None for auto-detect)
        batch_size: Batch size for inference to avoid OOM
    
    Returns:
        Predictions array (N, 2) with [SBP, DBP]
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    model.eval()
    
    # Handle single signal
    if ppg_signals.ndim == 1:
        ppg_signals = ppg_signals.reshape(1, -1)
    
    # Process in batches to avoid OOM
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(ppg_signals), batch_size):
            # Create batch
            batch_numpy = ppg_signals[i:i+batch_size]
            X_batch = torch.FloatTensor(batch_numpy).unsqueeze(1).to(device)
            
            # Predict
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    batch_preds = model(X_batch).float().cpu().numpy()
            else:
                batch_preds = model(X_batch).cpu().numpy()
                
            predictions.append(batch_preds)
    
    return np.vstack(predictions)


if __name__ == "__main__":
    # Test model creation
    print("Creating CNN model...")
    model = create_cnn_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Expected input shape: (batch, 1, 625)")
    print(f"Output shape: (batch, 2) [SBP, DBP]")
    
    # Test forward pass
    test_input = torch.randn(4, 1, 625)
    test_output = model(test_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {test_output.shape}")
