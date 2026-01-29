"""Example: Integrating AlphaFold structure predictions.

Demonstrates how to combine AlphaGenome sequence predictions with
AlphaFold structure predictions for comprehensive variant assessment.
"""

from alphagenome_plus.protein.alphafold_integration import (
    AlphaFoldStructureAnalyzer,
    ESMEmbeddingAnalyzer
)


def main():
    print("Structural Impact Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = AlphaFoldStructureAnalyzer()
    
    # Example: Analyze a missense variant in TP53
    uniprot_id = "P04637"  # TP53
    position = 175  # Known hotspot
    ref_aa = "R"  # Arginine
    alt_aa = "H"  # Histidine (R175H mutation)
    
    print(f"\nAnalyzing variant: {ref_aa}{position}{alt_aa} in {uniprot_id}")
    
    try:
        # Fetch and analyze structure
        impact = analyzer.analyze_variant_impact(
            uniprot_id=uniprot_id,
            position=position,
            ref_aa=ref_aa,
            alt_aa=alt_aa
        )
        
        print("\nStructural Impact Assessment:")
        print(f"  Variant ID: {impact.variant_id}")
        print(f"  pLDDT change: {impact.plddt_change:.2f}")
        print(f"  Buried residue: {impact.buried}")
        print(f"  Secondary structure: {impact.secondary_structure}")
        print(f"  Contacts disrupted: {impact.contacts_disrupted}")
        print(f"  Stability score (ΔΔG): {impact.stability_score:.2f} kcal/mol")
        
        # Interpret results
        print("\nInterpretation:")
        if impact.stability_score < -1.5:
            print("  ⚠️  Likely DESTABILIZING mutation")
        elif impact.stability_score > 1.0:
            print("  ✓ Likely STABILIZING mutation")
        else:
            print("  ~ Neutral or mild effect")
            
        if impact.buried and abs(impact.plddt_change) > 10:
            print("  ⚠️  Buried residue with significant confidence change")
            
        if impact.contacts_disrupted > 3:
            print(f"  ⚠️  Disrupts {impact.contacts_disrupted} contacts")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This example requires internet access to fetch AlphaFold structures")
    
    # ESM-2 embedding analysis
    print("\n" + "="*50)
    print("ESM-2 Embedding Analysis")
    
    esm_analyzer = ESMEmbeddingAnalyzer()
    
    # Example protein sequence (truncated for demonstration)
    sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
    
    print(f"\nSequence length: {len(sequence)}")
    print(f"Mutation: {ref_aa}{position}{alt_aa}")
    
    try:
        esm_analyzer.load_model()
        score = esm_analyzer.compute_mutation_score(
            sequence=sequence,
            position=min(position, len(sequence)-1),
            alt_aa=alt_aa
        )
        
        print(f"\nESM-2 mutation score: {score:.4f}")
        print("(Higher score indicates more deleterious effect)")
        
        if score > 0.5:
            print("⚠️  High impact mutation predicted")
        elif score > 0.3:
            print("~ Moderate impact")
        else:
            print("✓ Low impact")
            
    except ImportError:
        print("\nESM-2 not available. Install with: pip install fair-esm")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
