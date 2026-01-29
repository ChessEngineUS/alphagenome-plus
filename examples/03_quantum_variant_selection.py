"""Example: Quantum-inspired variant prioritization.

Demonstrates QAOA-based optimization for selecting the most informative
variants while minimizing redundancy.
"""

import numpy as np
from alphagenome_plus.quantum.qaoa_optimizer import (
    VariantPrioritizationQAOA,
    QAOAConfig
)


def main():
    # Simulate variant scores and correlations
    n_variants = 20
    
    # Pathogenicity scores (higher = more pathogenic)
    scores = np.random.beta(2, 5, n_variants) * 10
    
    # Correlation matrix (variants in LD have high correlation)
    correlations = np.eye(n_variants)
    for i in range(n_variants):
        for j in range(i+1, n_variants):
            if abs(i - j) < 3:  # Nearby variants are correlated
                corr = 0.8 * np.exp(-abs(i-j)/2)
                correlations[i, j] = corr
                correlations[j, i] = corr
    
    print("Variant Prioritization using QAOA")
    print("="*50)
    print(f"\nTotal variants: {n_variants}")
    print(f"Selecting top: 5 variants")
    
    # Initialize QAOA
    config = QAOAConfig(
        num_layers=3,
        max_iterations=100,
        optimizer='COBYLA',
        shots=1024
    )
    
    qaoa = VariantPrioritizationQAOA(config)
    
    print("\nRunning QAOA optimization...")
    selected_indices, objective_value = qaoa.optimize(
        scores=scores,
        correlations=correlations,
        k=5
    )
    
    print(f"\nSelected variant indices: {selected_indices}")
    print(f"Objective value: {objective_value:.4f}")
    
    print("\nSelected variant details:")
    for idx in selected_indices:
        print(f"  Variant {idx}: score = {scores[idx]:.4f}")
    
    # Compare with greedy selection
    print("\n" + "="*50)
    print("Comparison with Greedy Selection:")
    greedy_indices = np.argsort(scores)[-5:][::-1]
    greedy_objective = scores[greedy_indices].sum()
    
    print(f"Greedy selected: {greedy_indices.tolist()}")
    print(f"Greedy objective: {greedy_objective:.4f}")
    
    # Compute redundancy
    qaoa_redundancy = np.mean([correlations[i, j] 
                               for i in selected_indices 
                               for j in selected_indices if i < j])
    greedy_redundancy = np.mean([correlations[i, j] 
                                 for i in greedy_indices 
                                 for j in greedy_indices if i < j])
    
    print(f"\nQAOA redundancy: {qaoa_redundancy:.4f}")
    print(f"Greedy redundancy: {greedy_redundancy:.4f}")
    print(f"\nQAOA achieved {((greedy_redundancy - qaoa_redundancy)/greedy_redundancy*100):.1f}% "
          f"less redundancy!")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install Qiskit: pip install qiskit qiskit-aer")
