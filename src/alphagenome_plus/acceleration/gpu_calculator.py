"""GPU-accelerated variant effect prediction and calculation.

Provides CUDA-optimized implementations for batch variant scoring,
sequence encoding, and effect aggregation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU acceleration disabled.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class VariantBatch:
    """Batch of variants for GPU processing."""
    sequences: np.ndarray  # Shape: (batch_size, seq_len, 4) one-hot encoded
    positions: np.ndarray  # Shape: (batch_size,)
    reference_bases: List[str]
    alternate_bases: List[str]
    variant_ids: List[str]


class GPUSequenceEncoder:
    """GPU-accelerated DNA sequence encoding."""
    
    def __init__(self, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for GPU acceleration")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Nucleotide encoding mapping
        self.base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.index_to_base = {v: k for k, v in self.base_to_index.items()}
    
    def encode_sequences_batch(self, 
                               sequences: List[str],
                               max_length: Optional[int] = None) -> torch.Tensor:
        """Encode DNA sequences to one-hot tensors on GPU.
        
        Args:
            sequences: List of DNA sequences
            max_length: Maximum sequence length (pads/truncates)
            
        Returns:
            Tensor of shape (batch_size, seq_len, 5) with one-hot encoding
        """
        if max_length is None:
            max_length = max(len(s) for s in sequences)
        
        batch_size = len(sequences)
        
        # Initialize tensor
        encoded = torch.zeros(
            (batch_size, max_length, 5),
            dtype=torch.float32,
            device=self.device
        )
        
        # Encode each sequence
        for i, seq in enumerate(sequences):
            seq = seq.upper()
            seq_len = min(len(seq), max_length)
            
            for j, base in enumerate(seq[:seq_len]):
                idx = self.base_to_index.get(base, 4)  # 4 for unknown
                encoded[i, j, idx] = 1.0
        
        return encoded
    
    def reverse_complement_batch(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute reverse complement of one-hot encoded sequences.
        
        Args:
            sequences: Tensor of shape (batch, length, 5)
            
        Returns:
            Reverse complemented sequences
        """
        # Reverse along sequence dimension
        reversed_seq = torch.flip(sequences, dims=[1])
        
        # Complement: A<->T (0<->3), C<->G (1<->2), N stays N
        complemented = reversed_seq.clone()
        
        # Swap A and T
        a_channel = reversed_seq[:, :, 0].clone()
        t_channel = reversed_seq[:, :, 3].clone()
        complemented[:, :, 0] = t_channel
        complemented[:, :, 3] = a_channel
        
        # Swap C and G
        c_channel = reversed_seq[:, :, 1].clone()
        g_channel = reversed_seq[:, :, 2].clone()
        complemented[:, :, 1] = g_channel
        complemented[:, :, 2] = c_channel
        
        return complemented


class GPUVariantScorer:
    """GPU-accelerated variant effect scoring."""
    
    def __init__(self, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for GPU acceleration")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.encoder = GPUSequenceEncoder(device=str(self.device))
    
    def compute_position_weight_matrices(
        self,
        sequences: torch.Tensor,
        method: str = 'frequency'
    ) -> torch.Tensor:
        """Compute position weight matrices for sequences.
        
        Args:
            sequences: One-hot encoded sequences (batch, length, 5)
            method: 'frequency' or 'information_content'
            
        Returns:
            PWM tensor (length, 4)
        """
        # Sum across batch dimension
        counts = sequences[:, :, :4].sum(dim=0)  # (length, 4)
        
        if method == 'frequency':
            # Normalize to frequencies
            pwm = counts / (counts.sum(dim=1, keepdim=True) + 1e-8)
        elif method == 'information_content':
            # Information content (bits)
            frequencies = counts / (counts.sum(dim=1, keepdim=True) + 1e-8)
            # IC = freq * log2(freq / background)
            background = 0.25
            ic = frequencies * torch.log2((frequencies + 1e-8) / background)
            pwm = ic
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return pwm
    
    def score_variants_with_pwm(
        self,
        reference_seqs: torch.Tensor,
        alternate_seqs: torch.Tensor,
        pwm: torch.Tensor
    ) -> torch.Tensor:
        """Score variants using position weight matrix.
        
        Args:
            reference_seqs: Reference sequences (batch, length, 5)
            alternate_seqs: Alternate sequences (batch, length, 5)
            pwm: Position weight matrix (length, 4)
            
        Returns:
            Difference scores (batch,)
        """
        # Only use first 4 channels (ACGT)
        ref_seq = reference_seqs[:, :, :4]
        alt_seq = alternate_seqs[:, :, :4]
        
        # Expand PWM for batch
        pwm_expanded = pwm.unsqueeze(0).expand(ref_seq.shape[0], -1, -1)
        
        # Compute scores
        ref_scores = (ref_seq * pwm_expanded).sum(dim=[1, 2])
        alt_scores = (alt_seq * pwm_expanded).sum(dim=[1, 2])
        
        return alt_scores - ref_scores
    
    def compute_sequence_conservation(
        self,
        sequences: torch.Tensor,
        window_size: int = 7
    ) -> torch.Tensor:
        """Compute local sequence conservation scores.
        
        Args:
            sequences: One-hot sequences (batch, length, 5)
            window_size: Window for conservation calculation
            
        Returns:
            Conservation scores (batch, length)
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Use only ACGT channels
        seq_acgt = sequences[:, :, :4]
        
        # Compute Shannon entropy in sliding windows
        pad = window_size // 2
        padded = F.pad(seq_acgt.permute(0, 2, 1), (pad, pad), mode='replicate')
        
        # Unfold to get windows
        windows = padded.unfold(2, window_size, 1).permute(0, 2, 1, 3)
        
        # Compute entropy for each window
        # Sum across window to get base frequencies
        freq = windows.sum(dim=3) / window_size
        
        # Shannon entropy: -sum(p * log2(p))
        entropy = -(freq * torch.log2(freq + 1e-8)).sum(dim=2)
        
        # Conservation = max_entropy - entropy
        # max_entropy for 4 bases = 2.0
        conservation = 2.0 - entropy
        
        return conservation
    
    @torch.no_grad()
    def batch_variant_effect_prediction(
        self,
        variant_batch: VariantBatch,
        context_length: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Predict variant effects for a batch using GPU.
        
        Args:
            variant_batch: Batch of variants
            context_length: Sequence context length
            
        Returns:
            Dictionary of effect predictions
        """
        # Encode sequences
        ref_tensor = torch.from_numpy(variant_batch.sequences).float().to(self.device)
        
        batch_size = ref_tensor.shape[0]
        
        # Create alternate sequences
        alt_tensor = ref_tensor.clone()
        
        for i, (pos, ref_base, alt_base) in enumerate(
            zip(variant_batch.positions, 
                variant_batch.reference_bases,
                variant_batch.alternate_bases)
        ):
            # Clear reference base
            ref_idx = self.encoder.base_to_index.get(ref_base, 4)
            alt_idx = self.encoder.base_to_index.get(alt_base, 4)
            
            if pos < ref_tensor.shape[1]:
                alt_tensor[i, pos, :] = 0
                alt_tensor[i, pos, alt_idx] = 1.0
        
        # Compute PWM
        pwm = self.compute_position_weight_matrices(ref_tensor)
        
        # Score variants
        pwm_scores = self.score_variants_with_pwm(ref_tensor, alt_tensor, pwm)
        
        # Compute conservation
        ref_conservation = self.compute_sequence_conservation(ref_tensor)
        alt_conservation = self.compute_sequence_conservation(alt_tensor)
        
        # Conservation change at variant position
        conservation_change = torch.tensor([
            alt_conservation[i, pos] - ref_conservation[i, pos]
            for i, pos in enumerate(variant_batch.positions)
            if pos < ref_conservation.shape[1]
        ], device=self.device)
        
        # Compute GC content change
        ref_gc = self._compute_gc_content(ref_tensor)
        alt_gc = self._compute_gc_content(alt_tensor)
        gc_change = alt_gc - ref_gc
        
        return {
            'pwm_score_change': pwm_scores,
            'conservation_change': conservation_change,
            'gc_content_change': gc_change,
            'reference_conservation': ref_conservation.mean(dim=1),
            'alternate_conservation': alt_conservation.mean(dim=1)
        }
    
    def _compute_gc_content(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute GC content for sequences.
        
        Args:
            sequences: One-hot encoded (batch, length, 5)
            
        Returns:
            GC content (batch,)
        """
        # G is index 2, C is index 1
        gc_count = sequences[:, :, 1].sum(dim=1) + sequences[:, :, 2].sum(dim=1)
        total_count = sequences[:, :, :4].sum(dim=[1, 2])
        
        return gc_count / (total_count + 1e-8)


class EnsemblePredictor:
    """Ensemble predictions from multiple models/methods."""
    
    def __init__(self, device: Optional[str] = None):
        if TORCH_AVAILABLE:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = None
    
    def weighted_ensemble(
        self,
        predictions: List[Dict[str, Union[float, np.ndarray, torch.Tensor]]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Combine predictions using weighted ensemble.
        
        Args:
            predictions: List of prediction dictionaries from different models
            weights: Optional weights for each model (default: equal)
            
        Returns:
            Combined predictions
        """
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Combine predictions
        combined = {}
        
        # Get all keys
        all_keys = set()
        for pred in predictions:
            all_keys.update(pred.keys())
        
        for key in all_keys:
            values = []
            valid_weights = []
            
            for pred, weight in zip(predictions, weights):
                if key in pred:
                    val = pred[key]
                    
                    # Convert to numpy if tensor
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    
                    # Handle scalar or array
                    if isinstance(val, (int, float, np.number)):
                        values.append(float(val))
                        valid_weights.append(weight)
            
            if values:
                # Weighted average
                combined[key] = sum(v * w for v, w in zip(values, valid_weights))
        
        return combined
    
    def uncertainty_weighted_ensemble(
        self,
        predictions: List[Dict[str, float]],
        uncertainties: List[Dict[str, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """Ensemble using inverse uncertainty weighting.
        
        Args:
            predictions: List of predictions
            uncertainties: List of uncertainties (variance or std)
            
        Returns:
            Combined predictions with uncertainties
        """
        combined = {}
        
        # Get common keys
        all_keys = set(predictions[0].keys()) if predictions else set()
        for pred in predictions[1:]:
            all_keys &= set(pred.keys())
        
        for key in all_keys:
            values = [pred[key] for pred in predictions]
            uncerts = [unc.get(key, 1.0) for unc in uncertainties]
            
            # Weights = 1 / variance
            weights = [1.0 / (u ** 2 + 1e-8) for u in uncerts]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted mean
            mean = sum(v * w for v, w in zip(values, weights))
            
            # Combined uncertainty
            variance = 1.0 / total_weight if total_weight > 0 else 1.0
            
            combined[key] = (mean, np.sqrt(variance))
        
        return combined


def create_gpu_pipeline(batch_size: int = 32, 
                       device: Optional[str] = None) -> Dict:
    """Create GPU-accelerated variant analysis pipeline.
    
    Args:
        batch_size: Batch size for GPU processing
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary with pipeline components
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for GPU pipeline")
    
    encoder = GPUSequenceEncoder(device=device)
    scorer = GPUVariantScorer(device=device)
    ensemble = EnsemblePredictor(device=device)
    
    return {
        'encoder': encoder,
        'scorer': scorer,
        'ensemble': ensemble,
        'batch_size': batch_size,
        'device': scorer.device
    }
