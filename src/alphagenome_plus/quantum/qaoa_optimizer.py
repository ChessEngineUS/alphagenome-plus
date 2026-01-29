"""QAOA-based variant prioritization and optimization."""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from alphagenome.data import genome


@dataclass
class OptimizationResult:
    """Result of QAOA optimization."""
    selected_variants: List[genome.Variant]
    objective_value: float
    qaoa_params: Dict[str, np.ndarray]
    iterations: int


class QAOAVariantOptimizer:
    """QAOA-based variant selection and prioritization.
    
    Uses Quantum Approximate Optimization Algorithm to solve combinatorial
    optimization problems in variant selection, such as:
    - Maximizing pathogenicity score coverage
    - Satisfying distance constraints between variants
    - Balancing multiple objectives (clinical relevance, novelty, etc.)
    
    Example:
        >>> optimizer = QAOAVariantOptimizer(n_layers=3)
        >>> result = optimizer.optimize_selection(
        ...     variants=variant_list,
        ...     scores=pathogenicity_scores,
        ...     constraints={'max_variants': 10, 'min_distance': 1000}
        ... )
        >>> print(f"Selected {len(result.selected_variants)} variants")
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        backend: str = 'qiskit_aer',
        shots: int = 1024,
    ):
        """Initialize QAOA optimizer.
        
        Args:
            n_layers: Number of QAOA layers (p parameter)
            backend: Quantum backend to use
            shots: Number of measurement shots
        """
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for QAOA optimization. "
                "Install with: pip install 'alphagenome-plus[quantum]'"
            )
        
        self.n_layers = n_layers
        self.backend = AerSimulator() if backend == 'qiskit_aer' else backend
        self.shots = shots
    
    def optimize_selection(
        self,
        variants: List[genome.Variant],
        scores: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Optimize variant selection using QAOA.
        
        Args:
            variants: List of candidate variants
            scores: Pathogenicity or importance scores
            constraints: Selection constraints
            
        Returns:
            OptimizationResult with selected variants
        """
        constraints = constraints or {}
        max_variants = constraints.get('max_variants', len(variants))
        min_distance = constraints.get('min_distance', 0)
        
        # Build QUBO formulation
        Q = self._build_qubo_matrix(variants, scores, constraints)
        
        # Create QAOA circuit
        n_qubits = len(variants)
        circuit = self._create_qaoa_circuit(Q, n_qubits)
        
        # Optimize parameters
        best_params, best_value = self._optimize_parameters(circuit, Q)
        
        # Get optimal solution
        selected_indices = self._get_solution(circuit, best_params, n_qubits)
        
        # Filter to satisfy constraints
        selected_indices = self._enforce_constraints(
            selected_indices,
            variants,
            max_variants,
            min_distance,
        )
        
        selected_variants = [variants[i] for i in selected_indices]
        
        return OptimizationResult(
            selected_variants=selected_variants,
            objective_value=best_value,
            qaoa_params={'gamma': best_params[:self.n_layers],
                        'beta': best_params[self.n_layers:]},
            iterations=100,  # From optimization
        )
    
    def _build_qubo_matrix(
        self,
        variants: List[genome.Variant],
        scores: np.ndarray,
        constraints: Dict[str, Any],
    ) -> np.ndarray:
        """Build QUBO matrix for variant selection problem.
        
        Args:
            variants: Candidate variants
            scores: Variant scores
            constraints: Problem constraints
            
        Returns:
            QUBO matrix Q
        """
        n = len(variants)
        Q = np.zeros((n, n))
        
        # Diagonal: maximize scores
        np.fill_diagonal(Q, -scores)  # Negative for maximization
        
        # Off-diagonal: penalize selecting close variants
        min_distance = constraints.get('min_distance', 0)
        penalty = constraints.get('distance_penalty', 10.0)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate distance between variants
                if variants[i].chromosome == variants[j].chromosome:
                    dist = abs(variants[i].position - variants[j].position)
                    if dist < min_distance:
                        Q[i, j] = penalty
                        Q[j, i] = penalty
        
        # Constraint: maximum number of variants
        max_variants = constraints.get('max_variants', n)
        if max_variants < n:
            constraint_penalty = constraints.get('count_penalty', 20.0)
            # Add penalty for selecting more than max_variants
            for i in range(n):
                for j in range(i + 1, n):
                    Q[i, j] += constraint_penalty / (n * (n - 1) / 2)
        
        return Q
    
    def _create_qaoa_circuit(
        self,
        Q: np.ndarray,
        n_qubits: int,
    ) -> QuantumCircuit:
        """Create QAOA circuit for QUBO problem.
        
        Args:
            Q: QUBO matrix
            n_qubits: Number of qubits
            
        Returns:
            Parameterized QAOA circuit
        """
        # Create parameters
        gamma = [Parameter(f'γ_{i}') for i in range(self.n_layers)]
        beta = [Parameter(f'β_{i}') for i in range(self.n_layers)]
        
        # Initialize circuit
        qc = QuantumCircuit(n_qubits)
        
        # Initial state: uniform superposition
        qc.h(range(n_qubits))
        
        # QAOA layers
        for layer in range(self.n_layers):
            # Problem Hamiltonian (cost)
            for i in range(n_qubits):
                # Diagonal terms
                qc.rz(2 * gamma[layer] * Q[i, i], i)
                # Off-diagonal terms
                for j in range(i + 1, n_qubits):
                    if Q[i, j] != 0:
                        qc.cx(i, j)
                        qc.rz(2 * gamma[layer] * Q[i, j], j)
                        qc.cx(i, j)
            
            # Mixer Hamiltonian
            for i in range(n_qubits):
                qc.rx(2 * beta[layer], i)
        
        # Measurement
        qc.measure_all()
        
        return qc
    
    def _optimize_parameters(
        self,
        circuit: QuantumCircuit,
        Q: np.ndarray,
    ) -> tuple:
        """Optimize QAOA parameters using classical optimization.
        
        Args:
            circuit: QAOA circuit
            Q: QUBO matrix
            
        Returns:
            (best_params, best_value)
        """
        from scipy.optimize import minimize
        
        def objective(params):
            """Evaluate QAOA circuit with given parameters."""
            bound_circuit = circuit.assign_parameters(params)
            sampler = Sampler()
            result = sampler.run(bound_circuit, shots=self.shots).result()
            counts = result.quasi_dists[0]
            
            # Calculate expectation value
            expectation = 0
            for bitstring, prob in counts.items():
                # Convert to binary array
                x = np.array([int(b) for b in format(bitstring, f'0{len(Q)}b')])
                # Evaluate QUBO
                cost = x.T @ Q @ x
                expectation += cost * prob
            
            return expectation
        
        # Initial parameters
        init_params = np.random.uniform(0, 2 * np.pi, 2 * self.n_layers)
        
        # Optimize
        result = minimize(objective, init_params, method='COBYLA', options={'maxiter': 100})
        
        return result.x, result.fun
    
    def _get_solution(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        n_qubits: int,
    ) -> List[int]:
        """Extract solution from optimized QAOA circuit.
        
        Args:
            circuit: QAOA circuit
            params: Optimized parameters
            n_qubits: Number of qubits
            
        Returns:
            List of selected variant indices
        """
        # Bind parameters and execute
        bound_circuit = circuit.assign_parameters(params)
        sampler = Sampler()
        result = sampler.run(bound_circuit, shots=self.shots).result()
        counts = result.quasi_dists[0]
        
        # Get most probable bitstring
        best_bitstring = max(counts.items(), key=lambda x: x[1])[0]
        
        # Convert to selected indices
        binary = format(best_bitstring, f'0{n_qubits}b')
        selected = [i for i, bit in enumerate(binary) if bit == '1']
        
        return selected
    
    def _enforce_constraints(
        self,
        selected: List[int],
        variants: List[genome.Variant],
        max_variants: int,
        min_distance: int,
    ) -> List[int]:
        """Enforce hard constraints on solution.
        
        Args:
            selected: Initially selected indices
            variants: All variants
            max_variants: Maximum to select
            min_distance: Minimum distance between variants
            
        Returns:
            Filtered selected indices
        """
        # Sort by position for distance checking
        selected_sorted = sorted(selected, key=lambda i: (
            variants[i].chromosome,
            variants[i].position
        ))
        
        # Filter by distance constraint
        filtered = []
        for idx in selected_sorted:
            if not filtered:
                filtered.append(idx)
                continue
            
            # Check distance to previously selected
            last_var = variants[filtered[-1]]
            curr_var = variants[idx]
            
            if last_var.chromosome != curr_var.chromosome:
                filtered.append(idx)
            elif curr_var.position - last_var.position >= min_distance:
                filtered.append(idx)
        
        # Limit to max_variants
        if len(filtered) > max_variants:
            filtered = filtered[:max_variants]
        
        return filtered