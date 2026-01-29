"""Tests for protein folding integration."""

import pytest
import numpy as np
from alphagenome_plus.protein.alphafold_integration import (
    AlphaFoldIntegrator,
    ESMEmbeddingAnalyzer,
    VariantStructuralImpact
)


class TestAlphaFoldIntegrator:
    
    def test_initialization(self):
        integrator = AlphaFoldIntegrator(cache_structures=True)
        assert integrator.cache_structures is True
        assert len(integrator.structure_cache) == 0
    
    def test_structure_prediction(self):
        integrator = AlphaFoldIntegrator()
        sequence = "MKTIIALSYIF"
        
        structure = integrator._get_structure(sequence)
        
        assert 'sequence' in structure
        assert 'coordinates' in structure
        assert 'plddt' in structure
        assert len(structure['plddt']) == len(sequence)
    
    def test_mutation_application(self):
        integrator = AlphaFoldIntegrator()
        
        original = "MKTIIALSYIF"
        mutated = integrator._apply_mutation(original, 5, 'V')
        
        assert len(mutated) == len(original)
        assert mutated[4] == 'V'
        assert mutated[:4] == original[:4]
    
    def test_ddg_computation(self):
        integrator = AlphaFoldIntegrator()
        
        sequence = "MKTIIALSYIF"
        wt_structure = integrator._get_structure(sequence)
        mut_structure = integrator._get_structure(sequence)  # Same for simplicity
        
        ddg = integrator._compute_ddg(wt_structure, mut_structure, position=5)
        
        assert isinstance(ddg, float)
    
    def test_burial_score(self):
        integrator = AlphaFoldIntegrator()
        
        sequence = "MKTIIALSYIF"
        structure = integrator._get_structure(sequence)
        
        burial = integrator._compute_burial_score(structure, position=5)
        
        assert 0 <= burial <= 1
    
    def test_variant_structural_impact_prediction(self):
        integrator = AlphaFoldIntegrator()
        
        impact = integrator.predict_variant_structural_impact(
            variant_id='test_var',
            protein_sequence='MKTIIALSYIF',
            variant_position=5,
            reference_aa='I',
            alternate_aa='V'
        )
        
        assert isinstance(impact, VariantStructuralImpact)
        assert impact.variant_id == 'test_var'
        assert impact.position == 5
        assert 0 <= impact.structural_pathogenicity <= 1
    
    def test_batch_prediction(self):
        integrator = AlphaFoldIntegrator()
        
        sequence = "MKTIIALSYIF"
        variants = [
            ('var1', sequence, 3, 'T', 'A'),
            ('var2', sequence, 5, 'I', 'V'),
        ]
        
        results = integrator.batch_predict_structural_impacts(variants)
        
        assert len(results) == 2
        assert all(isinstance(r, VariantStructuralImpact) for r in results)
    
    def test_structural_pathogenicity_scoring(self):
        integrator = AlphaFoldIntegrator()
        
        # Destabilizing, buried, low confidence change
        score1 = integrator._compute_structural_pathogenicity(
            ddg=2.0,
            burial=0.8,
            plddt_change=-10,
            interface_disruption=0.5,
            alphagenome_scores=None
        )
        
        # Stabilizing, surface, high confidence
        score2 = integrator._compute_structural_pathogenicity(
            ddg=-1.0,
            burial=0.2,
            plddt_change=5,
            interface_disruption=0.1,
            alphagenome_scores=None
        )
        
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
        assert score1 > score2  # First should be more pathogenic


class TestESMEmbeddingAnalyzer:
    
    def test_initialization(self):
        analyzer = ESMEmbeddingAnalyzer()
        assert analyzer.model is not None
    
    def test_embedding_difference(self):
        analyzer = ESMEmbeddingAnalyzer()
        
        wt = "MKTIIALSYIF"
        mut = "MKTVALSYIF"  # I->V at position 4
        
        diff = analyzer.compute_embedding_difference(wt, mut)
        
        assert isinstance(diff, np.ndarray)
        assert len(diff) == 1280  # ESM-2 650M embedding size
    
    def test_functional_impact_prediction(self):
        analyzer = ESMEmbeddingAnalyzer()
        
        # Small change
        small_diff = np.random.randn(1280) * 0.1
        small_impact = analyzer.predict_functional_impact(small_diff)
        
        # Large change
        large_diff = np.random.randn(1280) * 10
        large_impact = analyzer.predict_functional_impact(large_diff)
        
        assert 0 <= small_impact <= 1
        assert 0 <= large_impact <= 1
        assert large_impact > small_impact
