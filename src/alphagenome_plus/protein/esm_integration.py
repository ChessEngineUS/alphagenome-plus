"""ESM-2 protein language model integration."""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class ESMEmbeddingResult:
    """ESM embedding result."""
    embeddings: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    sequence_length: int = 0


class ESMEmbeddings:
    """Extract ESM-2 embeddings for protein sequences.
    
    Example:
        >>> esm = ESMEmbeddings(model='esm2_t33_650M_UR50D')
        >>> result = esm.get_embeddings('MKTAYIAKQRQISFVKSHFSRQLE')
        >>> print(result.embeddings.shape)  # (seq_len, embed_dim)
    """
    
    def __init__(self, model: str = 'esm2_t33_650M_UR50D', device: str = 'cuda'):
        """Initialize ESM embeddings.
        
        Args:
            model: ESM model variant
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model
        self.device = device
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ESM model."""
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
                self.model_name
            )
            self.model.eval()
            if self.device == 'cuda':
                self.model = self.model.cuda()
        except ImportError:
            raise ImportError(
                "ESM is required. Install with: pip install fair-esm"
            )
    
    def get_embeddings(
        self,
        sequence: str,
        repr_layer: int = 33,
    ) -> ESMEmbeddingResult:
        """Extract embeddings for protein sequence.
        
        Args:
            sequence: Protein sequence
            repr_layer: Layer to extract representations from
            
        Returns:
            ESMEmbeddingResult
        """
        import torch
        
        # Prepare batch
        batch_converter = self.alphabet.get_batch_converter()
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        if self.device == 'cuda':
            batch_tokens = batch_tokens.cuda()
        
        # Extract embeddings
        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[repr_layer],
                return_contacts=False,
            )
        
        embeddings = results["representations"][repr_layer].cpu().numpy()[0]
        
        return ESMEmbeddingResult(
            embeddings=embeddings,
            sequence_length=len(sequence),
        )