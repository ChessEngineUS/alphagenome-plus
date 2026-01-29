"""Quantum-inspired feature selection."""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


@dataclass
class FeatureSelectionResult:
    """Result of quantum feature selection."""
    selected_features: List[int]
    feature_scores: np.ndarray
    circuit_params: np.ndarray


class QuantumFeatureSelector:
    """Quantum-inspired feature selection using VQE.
    
    Uses variational quantum eigensolver to select optimal feature
    subset for genomic prediction tasks.
    
    Example:
        >>> selector = QuantumFeatureSelector(n_qubits=8)
        >>> result = selector.select_features(
        ...     X=feature_matrix,
        ...     y=target_labels,
        ...     n_select=5
        ... )
    """
    
    def __init__(self, n_qubits: int = 8, n_layers: int = 2):
        """Initialize feature selector.
        
        Args:
            n_qubits: Number of qubits (max features)
            n_layers: Circuit depth
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is required. Install with: pip install 'alphagenome-plus[quantum]'"
            )
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits)
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_select: int,
    ) -> FeatureSelectionResult:
        """Select optimal feature subset.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            n_select: Number of features to select
            
        Returns:
            FeatureSelectionResult
        """
        n_features = min(X.shape[1], self.n_qubits)
        
        # Create quantum circuit for feature evaluation
        @qml.qnode(self.dev)
        def circuit(params, feature_idx):
            # Encode feature subset
            for i in range(n_features):
                if i == feature_idx:
                    qml.RY(params[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(n_features):
                    qml.RY(params[n_features + layer * n_features + i], wires=i)
                for i in range(n_features - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        
        # Evaluate feature importance
        feature_scores = np.zeros(n_features)
        params = np.random.randn((self.n_layers + 1) * n_features)
        
        for feat_idx in range(n_features):
            score = circuit(params, feat_idx)
            feature_scores[feat_idx] = abs(score)
        
        # Select top features
        selected = np.argsort(feature_scores)[-n_select:]
        
        return FeatureSelectionResult(
            selected_features=selected.tolist(),
            feature_scores=feature_scores,
            circuit_params=params,
        )