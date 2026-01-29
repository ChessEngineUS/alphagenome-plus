#!/usr/bin/env python3
"""Example: Protein folding integration with AlphaFold3.

Demonstrates combining AlphaGenome genomic predictions with
AlphaFold3 structural modeling to assess variant impacts.
"""

import numpy as np
from alphagenome_plus.protein.alphafold_integration import (
    AlphaFoldIntegrator,
    ESMEmbeddingAnalyzer
)


def main():
    print("="*70)
    print("Protein Folding Integration Example")
    print("="*70)
    
    # Example protein and variants
    protein_sequence = "MKTIIALSYIFCLVFADYKDDDDK"  # Example sequence
    
    variants = [
        ('var1', 5, 'A', 'V'),   # Position 5: A->V (hydrophobic to hydrophobic)
        ('var2', 10, 'F', 'L'),  # Position 10: F->L (aromatic to aliphatic)
        ('var3', 15, 'F', 'S'),  # Position 15: F->S (hydrophobic to polar)
    ]
    
    print(f"\nProtein sequence: {protein_sequence}")
    print(f"Length: {len(protein_sequence)} amino acids\n")
    
    # Initialize integrator
    print("[1] Initializing AlphaFold integrator...")
    integrator = AlphaFoldIntegrator(cache_structures=True)
    
    # Analyze each variant
    print("\n[2] Analyzing structural impacts...\n")
    
    for var_id, position, ref_aa, alt_aa in variants:
        print(f"Variant: {var_id} ({ref_aa}{position}{alt_aa})")
        print("-" * 50)
        
        # Mock AlphaGenome scores
        alphagenome_scores = {
            'rna_seq_effect': np.random.uniform(0, 1),
            'splicing_effect': np.random.uniform(0, 1),
            'pathogenicity': np.random.uniform(0, 1)
        }
        
        # Predict structural impact
        impact = integrator.predict_variant_structural_impact(
            variant_id=var_id,
            protein_sequence=protein_sequence,
            variant_position=position,
            reference_aa=ref_aa,
            alternate_aa=alt_aa,
            alphagenome_scores=alphagenome_scores
        )
        
        # Display results
        print(f"  ΔΔG (stability):        {impact.ddg_stability:+.2f} kcal/mol")
        print(f"  Burial score:           {impact.burial_score:.2f}")
        print(f"  Secondary structure:    {impact.secondary_structure}")
        print(f"  Binding site affected:  {impact.binding_site_affected}")
        print(f"  pLDDT change:           {impact.plddt_change:+.2f}")
        print(f"  Interface disruption:   {impact.interface_disruption:.2f}")
        print(f"  Structural pathogenicity: {impact.structural_pathogenicity:.3f}")
        print()
    
    # Batch analysis
    print("[3] Batch structural impact analysis...\n")
    
    batch_variants = [
        (var_id, protein_sequence, pos, ref, alt)
        for var_id, pos, ref, alt in variants
    ]
    
    batch_results = integrator.batch_predict_structural_impacts(batch_variants)
    
    # Sort by pathogenicity
    sorted_results = sorted(
        batch_results,
        key=lambda x: x.structural_pathogenicity,
        reverse=True
    )
    
    print("Variants ranked by structural pathogenicity:")
    print("Rank | Variant | Mutation | Pathogenicity | ΔΔG   | Burial")
    print("-" * 70)
    
    for rank, impact in enumerate(sorted_results, 1):
        mutation = f"{impact.reference_aa}{impact.position}{impact.alternate_aa}"
        print(
            f"{rank:4d} | {impact.variant_id:7s} | {mutation:8s} | "
            f"{impact.structural_pathogenicity:13.3f} | {impact.ddg_stability:+6.2f} | "
            f"{impact.burial_score:6.2f}"
        )
    
    # ESM-2 embedding analysis
    print("\n[4] ESM-2 embedding analysis...\n")
    
    esm_analyzer = ESMEmbeddingAnalyzer()
    
    for var_id, position, ref_aa, alt_aa in variants:
        # Create mutant sequence
        mutant_seq = list(protein_sequence)
        mutant_seq[position - 1] = alt_aa
        mutant_seq = ''.join(mutant_seq)
        
        # Compute embedding difference
        emb_diff = esm_analyzer.compute_embedding_difference(
            protein_sequence,
            mutant_seq
        )
        
        # Predict functional impact
        functional_impact = esm_analyzer.predict_functional_impact(emb_diff)
        
        print(f"{var_id}: Embedding Δ magnitude = {np.linalg.norm(emb_diff):.2f}, "
              f"Functional impact = {functional_impact:.3f}")
    
    print("\n" + "="*70)
    print("Protein folding integration complete!")
    print("="*70)


if __name__ == '__main__':
    main()
