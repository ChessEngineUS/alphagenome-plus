"""ML training pipeline for fine-tuning on AlphaGenome predictions.

Provides infrastructure for training downstream ML models on
AlphaGenome-derived features.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for ML training")


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-5
    dropout_rate: float = 0.2
    hidden_dims: List[int] = None
    device: str = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class VariantEffectDataset(Dataset):
    """PyTorch Dataset for variant effect prediction."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Dataset")
        
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
        if normalize:
            self.features = (self.features - self.features.mean(0)) / (self.features.std(0) + 1e-8)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class VariantEffectPredictor(nn.Module):
    """Neural network for variant effect prediction.
    
    Architecture:
        - Multi-layer perceptron with residual connections
        - Batch normalization and dropout
        - Multiple output heads for different effect types
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dims: Dict[str, int],
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.output_heads = list(output_dims.keys())
        
        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Output heads
        self.heads = nn.ModuleDict({
            head_name: nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, output_dim)
            )
            for head_name, output_dim in output_dims.items()
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Dictionary of predictions for each output head
        """
        # Shared encoding
        encoded = self.encoder(x)
        
        # Multiple output heads
        outputs = {
            head_name: head(encoded)
            for head_name, head in self.heads.items()
        }
        
        return outputs


class TrainingPipeline:
    """Complete training pipeline for variant effect models."""
    
    def __init__(self, config: TrainingConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")
        
        self.config = config
        self.model: Optional[VariantEffectPredictor] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_history: List[Dict[str, float]] = []
    
    def build_model(
        self,
        input_dim: int,
        output_dims: Dict[str, int]
    ):
        """Build model architecture.
        
        Args:
            input_dim: Number of input features
            output_dims: Dictionary mapping output head names to dimensions
        """
        self.model = VariantEffectPredictor(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dims=output_dims,
            dropout_rate=self.config.dropout_rate
        )
        
        self.model.to(self.config.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logging.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(
        self,
        train_dataset: VariantEffectDataset,
        val_dataset: Optional[VariantEffectDataset] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            loss_weights: Optional weights for each output head
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Loss functions
        criterion = nn.MSELoss()
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.config.device)
                batch_labels = batch_labels.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features)
                
                # Compute loss (assuming single output for simplicity)
                output_tensor = list(outputs.values())[0]
                loss = criterion(output_tensor.squeeze(), batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = None
            if val_dataset:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.config.device)
                        batch_labels = batch_labels.to(self.config.device)
                        
                        outputs = self.model(batch_features)
                        output_tensor = list(outputs.values())[0]
                        loss = criterion(output_tensor.squeeze(), batch_labels)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
            
            # Log progress
            epoch_stats = {'epoch': epoch, 'train_loss': train_loss}
            if val_loss is not None:
                epoch_stats['val_loss'] = val_loss
            
            self.training_history.append(epoch_stats)
            
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}: "
                    f"Train Loss = {train_loss:.4f}"
                    + (f", Val Loss = {val_loss:.4f}" if val_loss else "")
                )
        
        return self.training_history
    
    def save_model(self, path: str):
        """Save trained model.
        
        Args:
            path: Path to save model checkpoint
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path: str, input_dim: int, output_dims: Dict[str, int]):
        """Load trained model.
        
        Args:
            path: Path to model checkpoint
            input_dim: Input dimension (for model architecture)
            output_dims: Output dimensions (for model architecture)
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.build_model(input_dim, output_dims)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logging.info(f"Model loaded from {path}")
    
    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions on new data.
        
        Args:
            features: Input features [num_samples, input_dim]
        
        Returns:
            Dictionary of predictions for each output head
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        self.model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.config.device)
            outputs = self.model(features_tensor)
            
            predictions = {
                head_name: output.cpu().numpy()
                for head_name, output in outputs.items()
            }
        
        return predictions
