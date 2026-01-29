"""Extract and manipulate embeddings from AlphaGenome predictions.

Provides interfaces to extract latent representations for downstream ML tasks,
including fine-tuning, transfer learning, and feature engineering.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable
import torch
import torch.nn as nn
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from alphagenome.models import dna_client


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""
    embedding_dim: int = 768
    pooling_strategy: str = 'mean'  # 'mean', 'max', 'cls', 'attention'
    normalize: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class AlphaGenomeEmbeddingExtractor:
    """Extract embeddings from AlphaGenome model outputs.
    
    Converts AlphaGenome predictions into fixed-size vector representations
    suitable for downstream machine learning tasks.
    
    Args:
        model: AlphaGenome DNA client
        config: Embedding extraction configuration
    """
    
    def __init__(self, model: dna_client.DNAClient, 
                 config: Optional[EmbeddingConfig] = None):
        self.model = model
        self.config = config or EmbeddingConfig()
        
    def _pool_sequence(self, representations: np.ndarray, 
                      strategy: str = 'mean') -> np.ndarray:
        """Pool sequence-level representations to fixed size.
        
        Args:
            representations: Shape (seq_length, hidden_dim)
            strategy: Pooling strategy
            
        Returns:
            Pooled vector of shape (hidden_dim,)
        """
        if strategy == 'mean':
            return np.mean(representations, axis=0)
        elif strategy == 'max':
            return np.max(representations, axis=0)
        elif strategy == 'cls':
            return representations[0]  # First token
        elif strategy == 'attention':
            # Learned attention pooling
            attn_weights = np.softmax(representations @ representations[0], axis=0)
            return (representations.T @ attn_weights)
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
    
    def extract_from_prediction(self, prediction_output: Dict) -> np.ndarray:
        """Extract embedding from AlphaGenome prediction output.
        
        Args:
            prediction_output: Output from model.predict() or model.predict_variant()
            
        Returns:
            Embedding vector
        """
        # Extract relevant prediction tensors
        # AlphaGenome returns multimodal outputs - combine them
        embeddings = []
        
        if hasattr(prediction_output, 'rna_seq'):
            rna_embed = self._pool_sequence(
                prediction_output.rna_seq.data, 
                self.config.pooling_strategy
            )
            embeddings.append(rna_embed)
        
        if hasattr(prediction_output, 'chromatin_features'):
            chrom_embed = self._pool_sequence(
                prediction_output.chromatin_features.data,
                self.config.pooling_strategy  
            )
            embeddings.append(chrom_embed)
        
        # Concatenate multimodal embeddings
        combined = np.concatenate(embeddings)
        
        # Normalize if requested
        if self.config.normalize:
            combined = combined / (np.linalg.norm(combined) + 1e-8)
        
        return combined
    
    def extract_batch(self, intervals: List, 
                     ontology_terms: Optional[List[str]] = None,
                     output_types: Optional[List] = None) -> np.ndarray:
        """Extract embeddings for batch of genomic intervals.
        
        Args:
            intervals: List of genome.Interval objects
            ontology_terms: Optional tissue/cell type terms
            output_types: Types of predictions to include
            
        Returns:
            Matrix of embeddings, shape (n_intervals, embedding_dim)
        """
        embeddings = []
        
        for interval in intervals:
            # Get prediction
            outputs = self.model.predict(
                interval=interval,
                ontology_terms=ontology_terms or [],
                requested_outputs=output_types or [dna_client.OutputType.RNA_SEQ]
            )
            
            # Extract embedding
            embed = self.extract_from_prediction(outputs)
            embeddings.append(embed)
        
        return np.stack(embeddings)


class PyTorchEmbeddingAdapter(nn.Module):
    """PyTorch module wrapping AlphaGenome embeddings.
    
    Enables integration with PyTorch training pipelines and allows
    fine-tuning of embedding projections.
    
    Args:
        extractor: AlphaGenomeEmbeddingExtractor instance
        projection_dim: Output dimension after projection (None for no projection)
        freeze_extractor: Whether to freeze embedding extraction
    """
    
    def __init__(self, extractor: AlphaGenomeEmbeddingExtractor,
                 projection_dim: Optional[int] = None,
                 freeze_extractor: bool = True):
        super().__init__()
        self.extractor = extractor
        self.freeze_extractor = freeze_extractor
        
        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(extractor.config.embedding_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None
    
    def forward(self, prediction_outputs: List[Dict]) -> torch.Tensor:
        """Forward pass converting AlphaGenome outputs to tensors.
        
        Args:
            prediction_outputs: List of AlphaGenome prediction dicts
            
        Returns:
            Tensor of shape (batch_size, embedding_dim or projection_dim)
        """
        # Extract embeddings (numpy)
        with torch.set_grad_enabled(not self.freeze_extractor):
            embeddings = []
            for output in prediction_outputs:
                embed = self.extractor.extract_from_prediction(output)
                embeddings.append(embed)
            
            # Convert to torch tensor
            embed_tensor = torch.from_numpy(np.stack(embeddings)).float()
            embed_tensor = embed_tensor.to(self.extractor.config.device)
        
        # Apply projection if defined
        if self.projection is not None:
            embed_tensor = self.projection(embed_tensor)
        
        return embed_tensor


class JAXEmbeddingAdapter:
    """JAX interface for AlphaGenome embeddings.
    
    Provides functional interface for JAX-based training and optimization.
    
    Args:
        extractor: AlphaGenomeEmbeddingExtractor instance
    """
    
    def __init__(self, extractor: AlphaGenomeEmbeddingExtractor):
        self.extractor = extractor
    
    def extract_jax(self, prediction_outputs: List[Dict]) -> jnp.ndarray:
        """Extract embeddings as JAX array.
        
        Args:
            prediction_outputs: AlphaGenome prediction outputs
            
        Returns:
            JAX array of embeddings
        """
        embeddings = []
        for output in prediction_outputs:
            embed = self.extractor.extract_from_prediction(output)
            embeddings.append(embed)
        
        return jnp.array(np.stack(embeddings))
    
    def apply_projection(self, embeddings: jnp.ndarray, 
                        params: Dict) -> jnp.ndarray:
        """Apply learned projection to embeddings.
        
        Args:
            embeddings: Input embeddings
            params: Dictionary of projection parameters
            
        Returns:
            Projected embeddings
        """
        x = embeddings
        
        # Linear projection
        x = x @ params['weight'] + params['bias']
        
        # LayerNorm
        x = (x - jnp.mean(x, axis=-1, keepdims=True)) / (
            jnp.std(x, axis=-1, keepdims=True) + 1e-5
        )
        x = params['ln_scale'] * x + params['ln_bias']
        
        # ReLU
        x = jax.nn.relu(x)
        
        return x


class FineTuningDataset(torch.utils.data.Dataset):
    """PyTorch dataset for fine-tuning on genomic tasks.
    
    Args:
        intervals: List of genomic intervals
        labels: Target labels for supervised learning
        extractor: Embedding extractor
        transform: Optional data transformation
    """
    
    def __init__(self, intervals: List, labels: List,
                 extractor: AlphaGenomeEmbeddingExtractor,
                 transform: Optional[Callable] = None):
        self.intervals = intervals
        self.labels = labels
        self.extractor = extractor
        self.transform = transform
        
        # Pre-extract embeddings for efficiency
        self.embeddings = extractor.extract_batch(intervals)
    
    def __len__(self) -> int:
        return len(self.intervals)
    
    def __getitem__(self, idx: int):
        embed = self.embeddings[idx]
        label = self.labels[idx]
        
        if self.transform:
            embed = self.transform(embed)
        
        return torch.from_numpy(embed).float(), torch.tensor(label)


def create_downstream_classifier(input_dim: int, num_classes: int,
                                hidden_dims: List[int] = [512, 256]) -> nn.Module:
    """Create PyTorch classifier for downstream task.
    
    Args:
        input_dim: Input embedding dimension
        num_classes: Number of output classes
        hidden_dims: Hidden layer dimensions
        
    Returns:
        PyTorch sequential model
    """
    layers = []
    
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(nn.Linear(prev_dim, num_classes))
    
    return nn.Sequential(*layers)
