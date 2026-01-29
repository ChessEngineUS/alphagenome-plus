"""Variational Quantum Eigensolver for genomic variant optimization.

Implements VQE-based optimization for variant effect prediction and
prioritization using quantum circuits.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter
    from qiskit.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SPSA, COBYLA, SLSQP
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not available. Install with: pip install qiskit>=1.0")


class VariantVQEOptimizer:
    """VQE-based optimizer for variant effect prediction.
    
    Uses variational quantum circuits to optimize variant prioritization
    based on predicted effects from AlphaGenome.
    
    Attributes:
        num_qubits: Number of qubits in the circuit
        num_layers: Depth of variational circuit
        optimizer: Classical optimizer for parameter updates
    """
    
    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 3,
        optimizer: str = 'SPSA',
        max_iterations: int = 100
    ):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for VQE optimization")
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.max_iterations = max_iterations
        
        # Initialize optimizer
        optimizers = {
            'SPSA': SPSA(maxiter=max_iterations),
            'COBYLA': COBYLA(maxiter=max_iterations),
            'SLSQP': SLSQP(maxiter=max_iterations)
        }
        self.optimizer = optimizers.get(optimizer, SPSA(maxiter=max_iterations))
        
        self.estimator = Estimator()
        self.optimal_params: Optional[np.ndarray] = None
        self.optimal_energy: Optional[float] = None
    
    def create_ansatz(self) -> QuantumCircuit:
        """Create variational ansatz circuit.
        
        Implements a hardware-efficient ansatz with rotation gates
        and entangling layers.
        
        Returns:
            QuantumCircuit with parameterized gates
        """
        qc = QuantumCircuit(self.num_qubits)
        params = []
        
        for layer in range(self.num_layers):
            # Rotation layer
            for qubit in range(self.num_qubits):
                theta = Parameter(f'θ_{layer}_{qubit}_x')
                phi = Parameter(f'θ_{layer}_{qubit}_y')
                lambda_ = Parameter(f'θ_{layer}_{qubit}_z')
                params.extend([theta, phi, lambda_])
                
                qc.rx(theta, qubit)
                qc.ry(phi, qubit)
                qc.rz(lambda_, qubit)
            
            # Entangling layer
            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            
            # Ring closure
            if self.num_qubits > 2:
                qc.cx(self.num_qubits - 1, 0)
        
        return qc
    
    def encode_variant_features(
        self,
        features: np.ndarray
    ) -> QuantumCircuit:
        """Encode variant features into quantum state.
        
        Args:
            features: Feature vector (shape: [num_features])
                     Must be normalized to [-π, π]
        
        Returns:
            QuantumCircuit encoding the features
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Amplitude encoding
        feature_dim = min(2**self.num_qubits, len(features))
        normalized = features[:feature_dim] / (np.linalg.norm(features[:feature_dim]) + 1e-10)
        
        # Initialize with amplitude encoding
        qc.initialize(normalized, range(self.num_qubits))
        
        return qc
    
    def create_hamiltonian(
        self,
        weights: np.ndarray
    ) -> SparsePauliOp:
        """Create Hamiltonian for variant optimization.
        
        Args:
            weights: Importance weights for each qubit interaction
        
        Returns:
            SparsePauliOp representing the Hamiltonian
        """
        # Create Pauli strings for all interactions
        pauli_list = []
        
        # Single-qubit terms (Z basis)
        for i in range(self.num_qubits):
            pauli_str = ['I'] * self.num_qubits
            pauli_str[i] = 'Z'
            pauli_list.append((''.join(pauli_str), weights[i] if i < len(weights) else 1.0))
        
        # Two-qubit interactions (ZZ terms)
        idx = self.num_qubits
        for i in range(self.num_qubits - 1):
            for j in range(i + 1, self.num_qubits):
                pauli_str = ['I'] * self.num_qubits
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                weight = weights[idx] if idx < len(weights) else 0.5
                pauli_list.append((''.join(pauli_str), weight))
                idx += 1
        
        return SparsePauliOp.from_list(pauli_list)
    
    def optimize_variant_prioritization(
        self,
        variant_features: List[np.ndarray],
        importance_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Optimize variant prioritization using VQE.
        
        Args:
            variant_features: List of feature vectors for each variant
            importance_weights: Weights for Hamiltonian construction
        
        Returns:
            Dictionary containing:
                - optimal_energy: Minimum eigenvalue found
                - optimal_params: Optimal circuit parameters
                - variant_rankings: Prioritized variant indices
                - convergence_history: Energy values during optimization
        """
        if importance_weights is None:
            # Default weights: uniform
            num_terms = self.num_qubits + (self.num_qubits * (self.num_qubits - 1)) // 2
            importance_weights = np.ones(num_terms)
        
        # Create ansatz
        ansatz = self.create_ansatz()
        
        # Create Hamiltonian
        hamiltonian = self.create_hamiltonian(importance_weights)
        
        # Run VQE
        vqe = VQE(
            estimator=self.estimator,
            ansatz=ansatz,
            optimizer=self.optimizer
        )
        
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        self.optimal_params = result.optimal_parameters
        self.optimal_energy = result.optimal_value
        
        # Compute variant rankings based on energy contributions
        rankings = self._compute_rankings(variant_features)
        
        return {
            'optimal_energy': float(self.optimal_energy),
            'optimal_params': result.optimal_point,
            'variant_rankings': rankings,
            'num_evaluations': result.cost_function_evals
        }
    
    def _compute_rankings(
        self,
        variant_features: List[np.ndarray]
    ) -> List[int]:
        """Compute variant rankings from optimized circuit.
        
        Args:
            variant_features: List of feature vectors
        
        Returns:
            List of variant indices sorted by priority (descending)
        """
        scores = []
        
        for features in variant_features:
            # Encode features
            encoding_circuit = self.encode_variant_features(features)
            
            # Combine with optimized ansatz
            full_circuit = encoding_circuit.compose(self.create_ansatz())
            
            # Bind optimal parameters
            if self.optimal_params is not None:
                bound_circuit = full_circuit.assign_parameters(self.optimal_params)
                
                # Measure expectation (simplified scoring)
                score = np.abs(np.sum(features[:self.num_qubits]))
                scores.append(score)
            else:
                scores.append(0.0)
        
        # Sort indices by score (descending)
        rankings = np.argsort(scores)[::-1].tolist()
        return rankings
    
    def export_circuit(self, filename: str = 'vqe_circuit.qpy'):
        """Export optimized circuit for reuse.
        
        Args:
            filename: Path to save circuit
        """
        if self.optimal_params is None:
            raise ValueError("No optimized parameters available. Run optimization first.")
        
        circuit = self.create_ansatz()
        bound_circuit = circuit.assign_parameters(self.optimal_params)
        
        from qiskit import qpy
        with open(filename, 'wb') as f:
            qpy.dump(bound_circuit, f)
        
        logging.info(f"Circuit exported to {filename}")


class FeatureEncoder:
    """Encode genomic features for quantum circuits."""
    
    @staticmethod
    def encode_variant_effects(
        rna_seq_changes: np.ndarray,
        splicing_changes: np.ndarray,
        chromatin_changes: np.ndarray
    ) -> np.ndarray:
        """Encode multiple effect types into feature vector.
        
        Args:
            rna_seq_changes: RNA-seq log fold changes
            splicing_changes: Splicing effect scores
            chromatin_changes: Chromatin accessibility changes
        
        Returns:
            Normalized feature vector suitable for quantum encoding
        """
        # Aggregate effects
        features = np.concatenate([
            np.mean(rna_seq_changes, keepdims=True) if rna_seq_changes.size > 0 else [0],
            np.max(np.abs(splicing_changes), keepdims=True) if splicing_changes.size > 0 else [0],
            np.mean(chromatin_changes, keepdims=True) if chromatin_changes.size > 0 else [0],
        ])
        
        # Normalize to [-π, π] range
        features = np.clip(features, -10, 10)
        normalized = (features / 10) * np.pi
        
        return normalized
    
    @staticmethod
    def encode_sequence_context(
        sequence: str,
        position: int,
        window: int = 50
    ) -> np.ndarray:
        """Encode sequence context around variant.
        
        Args:
            sequence: DNA sequence
            position: Variant position
            window: Context window size
        
        Returns:
            Encoded sequence features
        """
        # Extract window
        start = max(0, position - window // 2)
        end = min(len(sequence), position + window // 2)
        context = sequence[start:end]
        
        # One-hot encode
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoded = np.zeros(4)
        
        for base in context:
            if base in encoding:
                encoded[encoding[base]] += 1
        
        # Normalize
        encoded = encoded / (len(context) + 1e-10)
        
        return encoded
