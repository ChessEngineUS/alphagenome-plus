"""Unit tests for quantum variant prioritization."""

import pytest
import numpy as np
from alphagenome_plus.quantum.variant_prioritization import (
    QAOAVariantPrioritizer,
    VariantScore,
    prioritize_variants_qaoa
)


class TestVariantScore:
    """Test VariantScore dataclass."""
    
    def test_creation(self):
        score = VariantScore(
            variant_id='rs123',
            pathogenicity=0.8,
            functional_impact=0.6,
            conservation=0.9
        )
        assert score.variant_id == 'rs123'
        assert score.pathogenicity == 0.8


class TestQAOAVariantPrioritizer:
    """Test QAOA variant prioritization."""
    
    @pytest.fixture
    def prioritizer(self):
        return QAOAVariantPrioritizer(n_qubits=4, n_layers=2)
    
    @pytest.fixture
    def sample_scores(self):
        return [
            VariantScore('var1', 0.9, 0.8, 0.7),
            VariantScore('var2', 0.3, 0.4, 0.5),
            VariantScore('var3', 0.7, 0.6, 0.8),
            VariantScore('var4', 0.5, 0.5, 0.5),
        ]
    
    def test_initialization(self, prioritizer):
        assert prioritizer.n_qubits == 4
        assert prioritizer.n_layers == 2
        assert prioritizer.optimal_params is None
    
    def test_cost_hamiltonian_shape(self, prioritizer, sample_scores):
        H_cost = prioritizer._cost_hamiltonian(sample_scores)
        expected_dim = 2**prioritizer.n_qubits
        assert H_cost.shape == (expected_dim, expected_dim)
    
    def test_mixer_hamiltonian_shape(self, prioritizer):
        H_mixer = prioritizer._mixer_hamiltonian()
        expected_dim = 2**prioritizer.n_qubits
        assert H_mixer.shape == (expected_dim, expected_dim)
        # Mixer should be symmetric
        assert np.allclose(H_mixer, H_mixer.T)
    
    def test_optimize(self, prioritizer, sample_scores):
        params, cost = prioritizer.optimize(sample_scores, max_iterations=10)
        
        # Check parameters were optimized
        assert params is not None
        assert len(params) == 2 * prioritizer.n_layers
        assert prioritizer.optimal_params is not None
        
        # Check cost is reasonable
        assert isinstance(cost, float)
    
    def test_get_ranking(self, prioritizer, sample_scores):
        # Must optimize first
        prioritizer.optimize(sample_scores, max_iterations=10)
        
        ranking = prioritizer.get_ranking(sample_scores)
        
        # Check ranking has all variants
        assert len(ranking) == len(sample_scores)
        
        # Check ranking is sorted descending by probability
        probs = [r[1] for r in ranking]
        assert probs == sorted(probs, reverse=True)
        
        # Check all probabilities are between 0 and 1
        assert all(0 <= p <= 1 for p in probs)
    
    def test_get_ranking_top_k(self, prioritizer, sample_scores):
        prioritizer.optimize(sample_scores, max_iterations=10)
        
        top_2 = prioritizer.get_ranking(sample_scores, top_k=2)
        
        assert len(top_2) == 2
        assert top_2[0][1] >= top_2[1][1]  # First has higher prob
    
    def test_get_ranking_before_optimize_raises(self, prioritizer, sample_scores):
        with pytest.raises(ValueError, match="Must call optimize"):
            prioritizer.get_ranking(sample_scores)
    
    def test_wrong_number_of_scores_raises(self, prioritizer):
        wrong_scores = [
            VariantScore('var1', 0.9, 0.8, 0.7),
            VariantScore('var2', 0.3, 0.4, 0.5),
        ]  # Only 2, need 4
        
        with pytest.raises(ValueError, match="Expected 4 scores"):
            prioritizer.optimize(wrong_scores)


class TestPrioritizeVariantsQAOA:
    """Test high-level prioritization function."""
    
    @pytest.fixture
    def variant_dicts(self):
        return [
            {'variant_id': 'rs1', 'pathogenicity': 0.9, 
             'functional_impact': 0.8, 'conservation': 0.7},
            {'variant_id': 'rs2', 'pathogenicity': 0.3,
             'functional_impact': 0.4, 'conservation': 0.5},
            {'variant_id': 'rs3', 'pathogenicity': 0.7,
             'functional_impact': 0.6, 'conservation': 0.8},
        ]
    
    def test_prioritize_all_variants(self, variant_dicts):
        result = prioritize_variants_qaoa(
            variant_dicts,
            n_layers=2,
            max_iterations=10
        )
        
        # Check all variants returned
        assert len(result) == len(variant_dicts)
        
        # Check QAOA priority added
        assert all('qaoa_priority' in v for v in result)
        
        # Check original fields preserved
        assert all('variant_id' in v for v in result)
        
        # Check sorted by priority
        priorities = [v['qaoa_priority'] for v in result]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_prioritize_top_k(self, variant_dicts):
        result = prioritize_variants_qaoa(
            variant_dicts,
            n_layers=2,
            max_iterations=10,
            top_k=2
        )
        
        assert len(result) == 2
    
    def test_missing_scores_use_defaults(self):
        variants_missing_scores = [
            {'variant_id': 'rs1'},
            {'variant_id': 'rs2', 'pathogenicity': 0.8},
            {'variant_id': 'rs3', 'conservation': 0.9},
        ]
        
        result = prioritize_variants_qaoa(
            variants_missing_scores,
            n_layers=2,
            max_iterations=5
        )
        
        # Should complete without error using defaults
        assert len(result) == 3
