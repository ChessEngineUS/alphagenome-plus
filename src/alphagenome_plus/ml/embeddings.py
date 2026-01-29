"""Advanced embedding extraction and manipulation for genomic sequences."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""
    embedding_dim: int = 512
    pooling_strategy: str = 'mean'  # 'mean', 'max', 'cls', 'attention'
    normalize: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class AlphaGenomeEmbeddingExtractor:
    """Extract and process embeddings from AlphaGenome predictions."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
    def extract_from_predictions(self, 
                                predictions: Dict[str, np.ndarray],
                                sequence_length: int) -> torch.Tensor:
        """Extract embeddings from AlphaGenome prediction outputs.
        
        Args:
            predictions: Dictionary of prediction arrays from AlphaGenome
            sequence_length: Length of the input sequence
            
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Stack different prediction modalities
        features = []
        
        for key, value in predictions.items():
            if isinstance(value, np.ndarray):
                # Flatten spatial dimensions if needed
                if len(value.shape) > 2:
                    value = value.reshape(value.shape[0], -1)
                features.append(torch.from_numpy(value))
        
        # Concatenate all features
        combined = torch.cat(features, dim=-1).to(self.device)
        
        # Apply pooling
        embeddings = self._pool_features(combined)
        
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
        return embeddings
    
    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to features."""
        if self.config.pooling_strategy == 'mean':
            return features.mean(dim=1)
        elif self.config.pooling_strategy == 'max':
            return features.max(dim=1)[0]
        elif self.config.pooling_strategy == 'cls':
            return features[:, 0, :]
        else:
            return features.mean(dim=1)


class ContrastiveLearningHead(nn.Module):
    """Contrastive learning head for genomic embeddings."""
    
    def __init__(self, input_dim: int, projection_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.projection(x), dim=-1)


class GenomicContrastiveLoss(nn.Module):
    """NT-Xent loss for genomic sequence contrastive learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two views.
        
        Args:
            z_i: First set of embeddings (batch_size, dim)
            z_j: Second set of embeddings (batch_size, dim)
            
        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute loss
        loss = nn.functional.cross_entropy(sim_matrix, labels)
        
        return loss


class VariantEffectPredictor(nn.Module):
    """Neural network for predicting variant effects from embeddings."""
    
    def __init__(self, 
                 embedding_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def compute_embedding_similarity(emb1: torch.Tensor, 
                                emb2: torch.Tensor,
                                metric: str = 'cosine') -> torch.Tensor:
    """Compute similarity between two sets of embeddings.
    
    Args:
        emb1: First embedding tensor (batch1, dim)
        emb2: Second embedding tensor (batch2, dim)
        metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        Similarity matrix (batch1, batch2)
    """
    if metric == 'cosine':
        emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=-1)
        emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=-1)
        return torch.mm(emb1_norm, emb2_norm.t())
    elif metric == 'euclidean':
        return -torch.cdist(emb1, emb2, p=2)
    elif metric == 'dot':
        return torch.mm(emb1, emb2.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")
