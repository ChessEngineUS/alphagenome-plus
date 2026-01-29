"""Quantum-inspired variant prioritization using QAOA.

This module implements quantum approximate optimization algorithm (QAOA)
for prioritizing variants based on predicted pathogenicity scores and
functional impact metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pennylane as qml
from pennylane import numpy as pnp


@dataclass
class VariantScore:
    """Container for variant scoring metrics."""
    variant_id: str
    pathogenicity: float  # 0-1 score
    functional_impact: float  # 0-1 score
    conservation: float  # 0-1 score
    combined_score: float = 0.0


class QAOAVariantPrioritizer:
    """QAOA-based variant prioritization system.
    
    Uses quantum optimization to find optimal variant ranking considering
    multiple competing objectives: pathogenicity, functional impact, and
    evolutionary conservation.
    
    Args:
        n_qubits: Number of qubits (should match number of variants)
        n_layers: Number of QAOA layers
        device: PennyLane device ('default.qubit' or simulator)
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 3, device: str = 'default.qubit'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)
        self.optimal_params = None
        
    def _cost_hamiltonian(self, scores: List[VariantScore]) -> np.ndarray:
        """Construct cost Hamiltonian encoding variant priorities.
        
        Returns:
            Hamiltonian matrix encoding optimization objective
        """
        # Construct weighted score matrix
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        
        for i, score in enumerate(scores):
            # Weight by combined importance
            weight = (0.5 * score.pathogenicity + 
                     0.3 * score.functional_impact +
                     0.2 * score.conservation)
            
            # Diagonal terms encode single-variant costs
            for state in range(2**self.n_qubits):
                if (state >> i) & 1:  # If qubit i is |1>
                    H[state, state] -= weight
                    
        return H
    
    def _mixer_hamiltonian(self) -> np.ndarray:
        """Construct mixer Hamiltonian for QAOA."""
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        
        for i in range(self.n_qubits):
            # X rotation mixer
            for state in range(2**self.n_qubits):
                flipped = state ^ (1 << i)
                H[state, flipped] += 1.0
                
        return H
    
    def _qaoa_circuit(self, params: np.ndarray, H_cost: np.ndarray, 
                      H_mixer: np.ndarray) -> qml.QNode:
        """Build QAOA quantum circuit.
        
        Args:
            params: Variational parameters [gamma_1, beta_1, ..., gamma_p, beta_p]
            H_cost: Cost Hamiltonian
            H_mixer: Mixer Hamiltonian
        """
        @qml.qnode(self.dev)
        def circuit():
            # Initialize in superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # QAOA layers
            for layer in range(self.n_layers):
                gamma = params[2*layer]
                beta = params[2*layer + 1]
                
                # Cost layer: e^(-i*gamma*H_cost)
                qml.ApproxTimeEvolution(H_cost, gamma, 1)
                
                # Mixer layer: e^(-i*beta*H_mixer) 
                qml.ApproxTimeEvolution(H_mixer, beta, 1)
            
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit
    
    def _objective(self, params: np.ndarray, scores: List[VariantScore]) -> float:
        """Compute QAOA objective function.
        
        Returns:
            Expected cost (to minimize)
        """
        H_cost = self._cost_hamiltonian(scores)
        H_mixer = self._mixer_hamiltonian()
        
        circuit = self._qaoa_circuit(params, H_cost, H_mixer)
        probs = circuit()
        
        # Compute expectation value
        expectation = 0.0
        for state, prob in enumerate(probs):
            expectation += prob * H_cost[state, state]
            
        return expectation
    
    def optimize(self, scores: List[VariantScore], 
                max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """Optimize QAOA parameters to find best variant ranking.
        
        Args:
            scores: List of variant scores to prioritize
            max_iterations: Maximum optimization iterations
            
        Returns:
            Tuple of (optimal parameters, final cost)
        """
        if len(scores) != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} scores, got {len(scores)}")
        
        # Initialize parameters randomly
        init_params = pnp.random.uniform(0, 2*np.pi, 2*self.n_layers, 
                                        requires_grad=True)
        
        # Gradient descent optimization
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        params = init_params
        
        for i in range(max_iterations):
            params = opt.step(lambda p: self._objective(p, scores), params)
            
            if i % 20 == 0:
                cost = self._objective(params, scores)
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        self.optimal_params = params
        final_cost = self._objective(params, scores)
        
        return params, final_cost
    
    def get_ranking(self, scores: List[VariantScore], 
                   top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get variant ranking based on optimized QAOA solution.
        
        Args:
            scores: Variant scores used in optimization
            top_k: Return only top k variants (None for all)
            
        Returns:
            List of (variant_id, probability) tuples, sorted by probability
        """
        if self.optimal_params is None:
            raise ValueError("Must call optimize() first")
        
        H_cost = self._cost_hamiltonian(scores)
        H_mixer = self._mixer_hamiltonian()
        
        circuit = self._qaoa_circuit(self.optimal_params, H_cost, H_mixer)
        probs = circuit()
        
        # Extract individual qubit marginals
        variant_probs = []
        for i, score in enumerate(scores):
            # Marginal probability for qubit i being |1>
            marginal = sum(probs[state] for state in range(2**self.n_qubits)
                          if (state >> i) & 1)
            variant_probs.append((score.variant_id, float(marginal)))
        
        # Sort by probability (descending)
        variant_probs.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            return variant_probs[:top_k]
        return variant_probs


def prioritize_variants_qaoa(variant_scores: List[Dict],
                            n_layers: int = 3,
                            max_iterations: int = 100,
                            top_k: Optional[int] = None) -> List[Dict]:
    """High-level function to prioritize variants using QAOA.
    
    Args:
        variant_scores: List of dicts with keys: variant_id, pathogenicity, 
                       functional_impact, conservation
        n_layers: Number of QAOA layers
        max_iterations: Maximum optimization iterations  
        top_k: Return only top k variants
        
    Returns:
        Sorted list of variant dicts with added 'qaoa_priority' field
    """
    # Convert to VariantScore objects
    scores = [
        VariantScore(
            variant_id=v['variant_id'],
            pathogenicity=v.get('pathogenicity', 0.5),
            functional_impact=v.get('functional_impact', 0.5),
            conservation=v.get('conservation', 0.5)
        )
        for v in variant_scores
    ]
    
    # Initialize and optimize QAOA
    n_variants = len(scores)
    prioritizer = QAOAVariantPrioritizer(n_qubits=n_variants, n_layers=n_layers)
    
    print(f"Optimizing QAOA for {n_variants} variants...")
    prioritizer.optimize(scores, max_iterations=max_iterations)
    
    # Get ranking
    ranking = prioritizer.get_ranking(scores, top_k=top_k)
    
    # Build output
    result = []
    for variant_id, priority in ranking:
        # Find original variant dict
        original = next(v for v in variant_scores if v['variant_id'] == variant_id)
        output = original.copy()
        output['qaoa_priority'] = priority
        result.append(output)
    
    return result
