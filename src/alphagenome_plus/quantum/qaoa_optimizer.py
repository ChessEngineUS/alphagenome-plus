"""QAOA-based optimization for variant prioritization and feature selection."""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not installed. Quantum optimization features disabled.")

try:
    import scipy.optimize as opt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class QAOAConfig:
    """Configuration for QAOA optimization."""
    num_layers: int = 3
    max_iterations: int = 100
    initial_point: Optional[List[float]] = None
    optimizer: str = 'COBYLA'
    shots: int = 1024


class VariantPrioritizationQAOA:
    """QAOA-based variant prioritization using quantum optimization.
    
    This implements a quantum approximate optimization algorithm for selecting
    the most important variants from a large set based on their predicted
    pathogenicity scores and pairwise correlations.
    """
    
    def __init__(self, config: QAOAConfig):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum optimization")
        
        self.config = config
        self.sampler = Sampler()
        
    def build_cost_hamiltonian(self, 
                               scores: np.ndarray,
                               correlations: np.ndarray,
                               k: int) -> Tuple[np.ndarray, float]:
        """Build cost Hamiltonian for variant selection.
        
        Args:
            scores: Pathogenicity scores for each variant (n_variants,)
            correlations: Correlation matrix between variants (n_variants, n_variants)
            k: Number of variants to select
            
        Returns:
            Hamiltonian matrix and offset
        """
        n = len(scores)
        
        # Construct QUBO matrix
        # Maximize: sum_i score_i * x_i - lambda * sum_{i<j} corr_ij * x_i * x_j
        # Subject to: sum_i x_i = k
        
        Q = np.zeros((n, n))
        
        # Linear terms (negative because we want to maximize scores)
        for i in range(n):
            Q[i, i] = -scores[i]
            
        # Quadratic terms (positive penalty for correlated variants)
        penalty = 2.0
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = penalty * correlations[i, j]
                Q[j, i] = penalty * correlations[i, j]
                
        # Cardinality constraint using penalty method
        constraint_penalty = 10.0
        for i in range(n):
            Q[i, i] += constraint_penalty * (1 - 2 * k)
            for j in range(n):
                if i != j:
                    Q[i, j] += constraint_penalty
                    
        return Q, 0.0
    
    def create_qaoa_circuit(self, 
                           hamiltonian: np.ndarray,
                           p: int) -> QuantumCircuit:
        """Create QAOA circuit for the given Hamiltonian.
        
        Args:
            hamiltonian: Cost Hamiltonian matrix
            p: Number of QAOA layers
            
        Returns:
            Parameterized quantum circuit
        """
        n = hamiltonian.shape[0]
        qc = QuantumCircuit(n)
        
        # Initial state: equal superposition
        qc.h(range(n))
        
        # QAOA layers
        beta = [Parameter(f'β_{i}') for i in range(p)]
        gamma = [Parameter(f'γ_{i}') for i in range(p)]
        
        for layer in range(p):
            # Cost Hamiltonian evolution
            for i in range(n):
                qc.rz(2 * gamma[layer] * hamiltonian[i, i], i)
                
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(hamiltonian[i, j]) > 1e-10:
                        qc.rzz(2 * gamma[layer] * hamiltonian[i, j], i, j)
            
            # Mixer Hamiltonian evolution
            for i in range(n):
                qc.rx(2 * beta[layer], i)
                
        qc.measure_all()
        return qc
    
    def optimize(self,
                scores: np.ndarray,
                correlations: np.ndarray,
                k: int) -> Tuple[List[int], float]:
        """Run QAOA optimization to select k variants.
        
        Args:
            scores: Pathogenicity scores (n_variants,)
            correlations: Correlation matrix (n_variants, n_variants)
            k: Number of variants to select
            
        Returns:
            Selected variant indices and objective value
        """
        hamiltonian, offset = self.build_cost_hamiltonian(scores, correlations, k)
        qc = self.create_qaoa_circuit(hamiltonian, self.config.num_layers)
        
        def cost_function(params):
            """Evaluate cost function for given parameters."""
            bound_circuit = qc.bind_parameters(params)
            job = self.sampler.run(bound_circuit, shots=self.config.shots)
            result = job.result()
            
            counts = result.quasi_dists[0]
            
            # Compute expectation value
            expectation = 0.0
            for bitstring, prob in counts.items():
                # Convert bitstring to binary array
                x = np.array([int(b) for b in format(bitstring, f'0{len(scores)}b')])
                
                # Compute cost
                cost = x @ hamiltonian @ x
                expectation += prob * cost
                
            return expectation + offset
        
        # Initialize parameters
        if self.config.initial_point is None:
            initial_point = np.random.uniform(0, 2*np.pi, 2*self.config.num_layers)
        else:
            initial_point = np.array(self.config.initial_point)
            
        # Optimize
        result = opt.minimize(cost_function,
                            initial_point,
                            method=self.config.optimizer,
                            options={'maxiter': self.config.max_iterations})
        
        # Get final measurement
        bound_circuit = qc.bind_parameters(result.x)
        job = self.sampler.run(bound_circuit, shots=self.config.shots)
        counts = job.result().quasi_dists[0]
        
        # Select most frequent bitstring with k ones
        valid_solutions = {}
        for bitstring, prob in counts.items():
            x = np.array([int(b) for b in format(bitstring, f'0{len(scores)}b')])
            if x.sum() == k:
                valid_solutions[bitstring] = prob
                
        if valid_solutions:
            best_solution = max(valid_solutions.items(), key=lambda x: x[1])[0]
            x_best = np.array([int(b) for b in format(best_solution, f'0{len(scores)}b')])
        else:
            # Fallback: select top k by score
            x_best = np.zeros(len(scores))
            top_k_indices = np.argsort(scores)[-k:]
            x_best[top_k_indices] = 1
            
        selected_indices = np.where(x_best == 1)[0].tolist()
        objective = scores[selected_indices].sum()
        
        return selected_indices, objective


class QuantumFeatureSelector:
    """Quantum-inspired feature selection for genomic data."""
    
    def __init__(self, n_features: int, n_qubits: int = None):
        self.n_features = n_features
        self.n_qubits = n_qubits or min(n_features, 20)
        
    def select_features(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       k: int) -> List[int]:
        """Select k most informative features using quantum-inspired algorithm.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            k: Number of features to select
            
        Returns:
            Indices of selected features
        """
        # Compute mutual information scores
        scores = self._compute_feature_scores(X, y)
        
        # Compute feature correlations
        correlations = np.corrcoef(X.T)
        
        # Use QAOA to select features
        qaoa = VariantPrioritizationQAOA(QAOAConfig(num_layers=2))
        selected_indices, _ = qaoa.optimize(scores, np.abs(correlations), k)
        
        return selected_indices
    
    def _compute_feature_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute feature importance scores."""
        from sklearn.feature_selection import mutual_info_classif
        return mutual_info_classif(X, y)
