#!/usr/bin/env python3
"""Example: Quantum optimization for variant prioritization.

Demonstrates using VQE and QAOA to prioritize clinically relevant variants
based on AlphaGenome predictions.
"""

import numpy as np
from alphagenome_plus.quantum.vqe_optimizer import VariantVQEOptimizer, FeatureEncoder
from alphagenome_plus.quantum.qaoa_prioritization import VariantPrioritizationQAOA, ClinicalRanker
from alphagenome.data import genome
from alphagenome.models import dna_client


def main():
    print("="*60)
    print("Quantum Variant Optimization Example")
    print("="*60)
    
    # Step 1: Get AlphaGenome predictions
    print("\n[1] Fetching AlphaGenome predictions...")
    
    API_KEY = 'YOUR_API_KEY'  # Replace with actual key
    # model = dna_client.create(API_KEY)
    
    # Mock variant data
    variants = [
        {'id': 'var1', 'rna_change': 2.5, 'splicing': 0.8, 'chromatin': 1.2},
        {'id': 'var2', 'rna_change': 1.8, 'splicing': 0.3, 'chromatin': 0.5},
        {'id': 'var3', 'rna_change': 3.2, 'splicing': 1.5, 'chromatin': 2.1},
        {'id': 'var4', 'rna_change': 0.5, 'splicing': 0.1, 'chromatin': 0.2},
        {'id': 'var5', 'rna_change': 2.1, 'splicing': 1.2, 'chromatin': 1.8},
    ]
    
    # Step 2: Encode features for quantum circuits
    print("\n[2] Encoding variant features...")
    
    encoder = FeatureEncoder()
    encoded_features = []
    
    for var in variants:
        features = encoder.encode_variant_effects(
            rna_seq_changes=np.array([var['rna_change']]),
            splicing_changes=np.array([var['splicing']]),
            chromatin_changes=np.array([var['chromatin']])
        )
        encoded_features.append(features)
        print(f"  {var['id']}: {features}")
    
    # Step 3: VQE Optimization
    print("\n[3] Running VQE optimization...")
    
    vqe_optimizer = VariantVQEOptimizer(
        num_qubits=4,
        num_layers=2,
        optimizer='SPSA',
        max_iterations=50
    )
    
    # Define importance weights (based on domain knowledge)
    importance_weights = np.array([1.0, 1.2, 0.8, 0.5, 0.6, 0.7])
    
    vqe_result = vqe_optimizer.optimize_variant_prioritization(
        variant_features=encoded_features,
        importance_weights=importance_weights
    )
    
    print(f"  Optimal energy: {vqe_result['optimal_energy']:.4f}")
    print(f"  VQE rankings: {vqe_result['variant_rankings']}")
    
    # Step 4: QAOA Clinical Prioritization
    print("\n[4] Running QAOA for clinical prioritization...")
    
    # Compute pathogenicity scores
    pathogenicity_scores = np.array([
        var['rna_change'] * 0.4 + var['splicing'] * 0.3 + var['chromatin'] * 0.3
        for var in variants
    ])
    
    qaoa_solver = VariantPrioritizationQAOA(
        num_layers=3,
        max_iterations=100,
        top_k=3
    )
    
    qaoa_result = qaoa_solver.solve_prioritization(
        pathogenicity_scores=pathogenicity_scores
    )
    
    print(f"  Selected variants: {qaoa_result['selected_variants']}")
    print(f"  Objective value: {qaoa_result['objective_value']:.4f}")
    
    # Step 5: Clinical Ranking with Multiple Criteria
    print("\n[5] Clinical ranking with ClinVar integration...")
    
    ranker = ClinicalRanker()
    
    variant_ids = [var['id'] for var in variants]
    alphagenome_scores = {var['id']: pathogenicity_scores[i] / 5.0 
                          for i, var in enumerate(variants)}
    clinvar_scores = {'var1': 0.8, 'var2': 0.3, 'var3': 0.9, 'var4': 0.1, 'var5': 0.7}
    pop_frequencies = {'var1': 0.001, 'var2': 0.05, 'var3': 0.0001, 'var4': 0.1, 'var5': 0.002}
    
    ranked_variants = ranker.rank_variants(
        variant_ids=variant_ids,
        alphagenome_scores=alphagenome_scores,
        clinvar_scores=clinvar_scores,
        population_frequencies=pop_frequencies,
        inheritance_pattern='autosomal_dominant'
    )
    
    print("\n  Clinical Prioritization Results:")
    print("  Rank | Variant | Composite Score")
    print("  " + "-"*40)
    for rank, (vid, score) in enumerate(ranked_variants, 1):
        print(f"  {rank:4d} | {vid:7s} | {score:.4f}")
    
    # Step 6: Export optimized circuit
    print("\n[6] Exporting optimized VQE circuit...")
    vqe_optimizer.export_circuit('optimized_vqe_circuit.qpy')
    print("  Circuit saved to: optimized_vqe_circuit.qpy")
    
    print("\n" + "="*60)
    print("Quantum optimization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
