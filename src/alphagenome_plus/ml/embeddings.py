"""Feature extraction and embedding generation."""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alphagenome.data import genome
from alphagenome.models import dna_client


@dataclass
class SequenceEmbedding:
    """Container for sequence embeddings."""
    embeddings: np.ndarray
    interval: genome.Interval
    layer: str
    metadata: Dict[str, Any]


class FeatureExtractor:
    """Extract features and embeddings from AlphaGenome predictions.
    
    Enables downstream ML tasks by extracting intermediate representations
    from AlphaGenome's predictions.
    
    Example:
        >>> extractor = FeatureExtractor(api_key='YOUR_KEY')
        >>> embeddings = extractor.get_sequence_embeddings(
        ...     interval=genome.Interval('chr22', 36200000, 36210000),
        ...     layer='intermediate'
        ... )
        >>> print(embeddings.shape)  # (sequence_length, embedding_dim)
    """
    
    def __init__(self, api_key: str):
        """Initialize feature extractor.
        
        Args:
            api_key: AlphaGenome API key
        """
        self.model = dna_client.create(api_key)
    
    def get_sequence_embeddings(
        self,
        interval: genome.Interval,
        ontology_terms: Optional[List[str]] = None,
        layer: str = 'intermediate',
    ) -> SequenceEmbedding:
        """Extract sequence embeddings for genomic interval.
        
        Args:
            interval: Genomic interval
            ontology_terms: Optional tissue/cell type context
            layer: Which layer to extract ('intermediate', 'final', 'attention')
            
        Returns:
            SequenceEmbedding object
        """
        # Make prediction to get outputs
        outputs = self.model.predict(
            interval=interval,
            ontology_terms=ontology_terms or ['UBERON:0000948'],  # Heart default
            requested_outputs=[dna_client.OutputType.RNA_SEQ],
        )
        
        # Extract embeddings from outputs
        # Note: This is a simplified example - actual implementation would
        # need to access model internals or use dedicated embedding endpoints
        rna_seq = outputs.rna_seq
        
        # Convert to numpy array and generate pseudo-embeddings
        # In real implementation, this would extract actual model embeddings
        embeddings = self._generate_embeddings_from_predictions(rna_seq, layer)
        
        return SequenceEmbedding(
            embeddings=embeddings,
            interval=interval,
            layer=layer,
            metadata={'ontology_terms': ontology_terms or []},
        )
    
    def _generate_embeddings_from_predictions(
        self,
        predictions: Any,
        layer: str,
    ) -> np.ndarray:
        """Generate embeddings from predictions.
        
        Note: Placeholder implementation. Real version would access
        model internals or use dedicated API endpoints.
        
        Args:
            predictions: Model predictions
            layer: Layer to extract from
            
        Returns:
            Embedding array
        """
        # Convert predictions to array
        if hasattr(predictions, 'values'):
            values = np.array(predictions.values)
        else:
            values = np.array(predictions)
        
        # Generate pseudo-embeddings based on layer
        embedding_dim = {'intermediate': 512, 'final': 256, 'attention': 128}[layer]
        
        # Reshape/project to embedding dimension
        seq_len = len(values)
        embeddings = np.random.randn(seq_len, embedding_dim)  # Placeholder
        
        return embeddings
    
    def extract_variant_features(
        self,
        variant: genome.Variant,
        interval: genome.Interval,
        ontology_terms: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract feature vectors for variant effect prediction.
        
        Args:
            variant: Variant to analyze
            interval: Genomic context
            ontology_terms: Tissue/cell type context
            
        Returns:
            Dictionary of feature vectors
        """
        # Get predictions for reference and alternate
        outputs = self.model.predict_variant(
            interval=interval,
            variant=variant,
            ontology_terms=ontology_terms or ['UBERON:0000948'],
            requested_outputs=[
                dna_client.OutputType.RNA_SEQ,
                dna_client.OutputType.CAGE,
            ],
        )
        
        # Extract features
        features = {
            'rna_seq_ref': np.array(outputs.reference.rna_seq.values),
            'rna_seq_alt': np.array(outputs.alternate.rna_seq.values),
            'rna_seq_diff': np.array(outputs.alternate.rna_seq.values) - 
                           np.array(outputs.reference.rna_seq.values),
        }
        
        if hasattr(outputs.reference, 'cage'):
            features['cage_ref'] = np.array(outputs.reference.cage.values)
            features['cage_alt'] = np.array(outputs.alternate.cage.values)
            features['cage_diff'] = features['cage_alt'] - features['cage_ref']
        
        return features


if TORCH_AVAILABLE:
    class EmbeddingModel(nn.Module):
        """PyTorch model for embedding-based prediction tasks.
        
        Example:
            >>> model = EmbeddingModel(embedding_dim=512, hidden_dim=256, output_dim=1)
            >>> embeddings = torch.randn(32, 100, 512)  # batch, seq, embed
            >>> predictions = model(embeddings)
        """
        
        def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int = 256,
            output_dim: int = 1,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            """Initialize embedding model.
            
            Args:
                embedding_dim: Input embedding dimension
                hidden_dim: Hidden layer dimension
                output_dim: Output dimension
                num_layers: Number of transformer layers
                dropout: Dropout rate
            """
            super().__init__()
            
            self.embedding_dim = embedding_dim
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            
            # Prediction head
            self.prediction_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        
        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            """Forward pass.
            
            Args:
                embeddings: Input embeddings (batch, seq_len, embedding_dim)
                
            Returns:
                Predictions (batch, output_dim)
            """
            # Apply transformer
            transformed = self.transformer(embeddings)
            
            # Global average pooling
            pooled = transformed.mean(dim=1)
            
            # Prediction
            output = self.prediction_head(pooled)
            
            return output
else:
    class EmbeddingModel:
        """Placeholder when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for EmbeddingModel. "
                "Install with: pip install torch"
            )