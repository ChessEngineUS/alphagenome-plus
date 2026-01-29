"""Tests for quantum optimization modules."""

import pytest
import numpy as np
from alphagenome_plus.quantum.vqe_optimizer import (
    VariantVQEOptimizer,
    FeatureEncoder
)
from alphagenome_plus.quantum.qaoa_prioritization import (
    VariantPrioritizationQAOA,
    ClinicalRanker
)


try:
    import qiskit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestVQEOptimizer:
    
    def test_initialization(self):
        optimizer = VariantVQEOptimizer(
            num_qubits=4,
            num_layers=2,
            optimizer='SPSA'
        )
        assert optimizer.num_qubits == 4
        assert optimizer.num_layers == 2
    
    def test_ansatz_creation(self):
        optimizer = VariantVQEOptimizer(num_qubits=3, num_layers=2)
        ansatz = optimizer.create_ansatz()
        
        assert ansatz.num_qubits == 3
        assert len(ansatz.parameters) > 0
    
    def test_feature_encoding(self):
        optimizer = VariantVQEOptimizer(num_qubits=4)
        features = np.random.randn(8)
        
        encoding_circuit = optimizer.encode_variant_features(features)
        assert encoding_circuit.num_qubits == 4
    
    def test_hamiltonian_creation(self):
        optimizer = VariantVQEOptimizer(num_qubits=3)
        weights = np.ones(10)
        
        hamiltonian = optimizer.create_hamiltonian(weights)
        assert hamiltonian is not None
    
    def test_optimization_workflow(self):
        optimizer = VariantVQEOptimizer(
            num_qubits=3,
            num_layers=1,
            max_iterations=10
        )
        
        variant_features = [
            np.random.randn(4),
            np.random.randn(4),
            np.random.randn(4)
        ]
        
        result = optimizer.optimize_variant_prioritization(
            variant_features=variant_features
        )
        
        assert 'optimal_energy' in result
        assert 'optimal_params' in result
        assert 'variant_rankings' in result
        assert len(result['variant_rankings']) == 3


class TestFeatureEncoder:
    
    def test_variant_effect_encoding(self):
        encoder = FeatureEncoder()
        
        rna = np.array([1.5, 2.0, 1.2])
        splicing = np.array([0.8, 0.5])
        chromatin = np.array([1.0, 1.5, 2.0, 1.2])
        
        features = encoder.encode_variant_effects(rna, splicing, chromatin)
        
        assert len(features) == 3
        assert np.all(np.abs(features) <= np.pi)
    
    def test_sequence_context_encoding(self):
        encoder = FeatureEncoder()
        
        sequence = "ATCGATCGATCG" * 10
        position = 50
        
        context_features = encoder.encode_sequence_context(sequence, position)
        
        assert len(context_features) == 4
        assert np.all(context_features >= 0)
        assert np.all(context_features <= 1)
        assert np.isclose(np.sum(context_features), 1.0)


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not installed")
class TestQAOAPrioritization:
    
    def test_qubo_formulation(self):
        solver = VariantPrioritizationQAOA(top_k=3)
        
        pathogenicity = np.array([0.8, 0.3, 0.9, 0.5, 0.7])
        
        qp = solver.formulate_qubo(pathogenicity)
        
        assert qp is not None
        assert len(qp.variables) == 5
    
    def test_solve_prioritization(self):
        solver = VariantPrioritizationQAOA(
            num_layers=2,
            max_iterations=20,
            top_k=2
        )
        
        pathogenicity = np.array([0.8, 0.3, 0.9, 0.5])
        
        result = solver.solve_prioritization(pathogenicity)
        
        assert 'selected_variants' in result
        assert len(result['selected_variants']) <= 2
        assert 'objective_value' in result
    
    def test_interaction_matrix_computation(self):
        solver = VariantPrioritizationQAOA()
        
        n = 4
        distances = np.random.uniform(0, 1e6, (n, n))
        pathways = np.random.randint(0, 5, (n, n))
        
        interactions = solver.compute_variant_interactions(distances, pathways)
        
        assert interactions.shape == (n, n)
        assert np.allclose(interactions, interactions.T)  # Symmetry


class TestClinicalRanker:
    
    def test_variant_ranking(self):
        ranker = ClinicalRanker()
        
        variant_ids = ['var1', 'var2', 'var3']
        alphagenome_scores = {'var1': 0.8, 'var2': 0.3, 'var3': 0.9}
        clinvar_scores = {'var1': 0.7, 'var2': 0.4, 'var3': 0.85}
        pop_freqs = {'var1': 0.001, 'var2': 0.05, 'var3': 0.0001}
        
        ranked = ranker.rank_variants(
            variant_ids=variant_ids,
            alphagenome_scores=alphagenome_scores,
            clinvar_scores=clinvar_scores,
            population_frequencies=pop_freqs
        )
        
        assert len(ranked) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranked)
        
        # Check descending order
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_composite_score_computation(self):
        ranker = ClinicalRanker()
        
        score = ranker._compute_composite_score(
            variant_id='var1',
            alphagenome_score=0.8,
            clinvar_score=0.7,
            pop_freq=0.001,
            inheritance='autosomal_dominant'
        )
        
        assert 0 <= score <= 1
