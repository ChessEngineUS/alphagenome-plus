"""Integration tests for AlphaGenome-Plus."""

import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock

from alphagenome.data import genome
from alphagenome_plus.batch import BatchVariantProcessor, BatchConfig
from alphagenome_plus.cache import PredictionCache
from alphagenome_plus.ml.embeddings import (
    AlphaGenomeEmbeddingExtractor,
    VariantEffectPredictor,
    EmbeddingConfig
)


class TestBatchProcessing:
    """Test batch variant processing."""
    
    @pytest.mark.asyncio
    async def test_single_variant_processing(self, mock_api_key):
        """Test processing a single variant."""
        processor = BatchVariantProcessor(api_key=mock_api_key)
        
        interval = genome.Interval(
            chromosome='chr22',
            start=35677410,
            end=36725986
        )
        
        variant = genome.Variant(
            chromosome='chr22',
            position=36201698,
            reference_bases='A',
            alternate_bases='C'
        )
        
        # Mock the API call
        with patch.object(processor, '_call_api', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = Mock()
            
            result = await processor.process_single_variant(
                interval=interval,
                variant=variant,
                ontology_terms=['UBERON:0001157']
            )
            
            assert result is not None
            mock_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_cache(self, mock_api_key, tmp_path):
        """Test batch processing with caching."""
        cache = PredictionCache(cache_dir=str(tmp_path))
        
        config = BatchConfig(
            max_concurrent=5,
            use_cache=True
        )
        
        processor = BatchVariantProcessor(
            api_key=mock_api_key,
            config=config,
            cache=cache
        )
        
        interval = genome.Interval(
            chromosome='chr22',
            start=35677410,
            end=36725986
        )
        
        variants = [
            genome.Variant(
                chromosome='chr22',
                position=36201698 + i * 1000,
                reference_bases='A',
                alternate_bases='T'
            )
            for i in range(10)
        ]
        
        with patch.object(processor, '_call_api', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = Mock()
            
            # First call
            results1 = await processor.process_variants(
                interval=interval,
                variants=variants[:5]
            )
            
            # Second call (should use cache for overlapping variants)
            results2 = await processor.process_variants(
                interval=interval,
                variants=variants
            )
            
            assert len(results1) == 5
            assert len(results2) == 10
            
            # Check cache was used
            assert cache.get_stats()['hits'] > 0


class TestMLIntegration:
    """Test ML integration features."""
    
    def test_embedding_extraction(self):
        """Test embedding extraction from predictions."""
        config = EmbeddingConfig(
            embedding_dim=512,
            pooling_strategy='mean',
            device='cpu'
        )
        
        extractor = AlphaGenomeEmbeddingExtractor(config)
        
        # Mock predictions
        predictions = {
            'rna_seq': np.random.randn(10, 32),
            'chip_seq': np.random.randn(10, 16)
        }
        
        embeddings = extractor.extract_from_predictions(
            predictions=predictions,
            sequence_length=131072
        )
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] <= config.embedding_dim
    
    def test_variant_classifier(self):
        """Test variant effect classifier."""
        model = VariantEffectPredictor(
            embedding_dim=512,
            hidden_dims=[256, 128],
            num_classes=3,
            dropout=0.3
        )
        
        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, 512)
        
        outputs = model(x)
        
        assert outputs.shape == (batch_size, 3)
        assert torch.all(torch.isfinite(outputs))
    
    def test_embedding_similarity(self):
        """Test embedding similarity computation."""
        from alphagenome_plus.ml.embeddings import compute_embedding_similarity
        
        emb1 = torch.randn(10, 128)
        emb2 = torch.randn(5, 128)
        
        # Cosine similarity
        sim_cosine = compute_embedding_similarity(emb1, emb2, metric='cosine')
        assert sim_cosine.shape == (10, 5)
        assert torch.all(sim_cosine >= -1) and torch.all(sim_cosine <= 1)
        
        # Euclidean distance
        sim_euclidean = compute_embedding_similarity(emb1, emb2, metric='euclidean')
        assert sim_euclidean.shape == (10, 5)


class TestQuantumOptimization:
    """Test quantum optimization features."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required"
    )
    def test_qaoa_variant_selection(self):
        """Test QAOA-based variant prioritization."""
        from alphagenome_plus.quantum.qaoa_optimizer import (
            VariantPrioritizationQAOA,
            QAOAConfig
        )
        
        # Small problem for testing
        n_variants = 10
        scores = np.random.rand(n_variants) * 10
        correlations = np.eye(n_variants) * 0.1
        
        config = QAOAConfig(
            num_layers=1,
            max_iterations=10,
            shots=128
        )
        
        qaoa = VariantPrioritizationQAOA(config)
        
        selected_indices, objective = qaoa.optimize(
            scores=scores,
            correlations=correlations,
            k=3
        )
        
        assert len(selected_indices) == 3
        assert objective > 0
        assert all(0 <= idx < n_variants for idx in selected_indices)
    
    def test_hamiltonian_construction(self):
        """Test cost Hamiltonian construction."""
        from alphagenome_plus.quantum.qaoa_optimizer import VariantPrioritizationQAOA
        
        qaoa = VariantPrioritizationQAOA(QAOAConfig())
        
        scores = np.array([1.0, 2.0, 3.0])
        correlations = np.eye(3)
        
        H, offset = qaoa.build_cost_hamiltonian(scores, correlations, k=2)
        
        assert H.shape == (3, 3)
        assert np.allclose(H, H.T)  # Symmetric


class TestProteinIntegration:
    """Test protein structure integration."""
    
    @pytest.mark.skip(reason="Requires network access")
    def test_alphafold_structure_fetch(self):
        """Test fetching AlphaFold structure."""
        from alphagenome_plus.protein.alphafold_integration import (
            AlphaFoldStructureAnalyzer
        )
        
        analyzer = AlphaFoldStructureAnalyzer()
        structure = analyzer.fetch_structure("P04637")  # TP53
        
        assert structure is not None
        assert 'plddt' in structure
    
    def test_stability_score_computation(self):
        """Test protein stability score calculation."""
        from alphagenome_plus.protein.alphafold_integration import (
            AlphaFoldStructureAnalyzer
        )
        
        analyzer = AlphaFoldStructureAnalyzer()
        
        # Buried hydrophobic to charged (destabilizing)
        score1 = analyzer._compute_stability_score('L', 'K', buried=True)
        assert score1 < 0
        
        # Surface charge change (mild effect)
        score2 = analyzer._compute_stability_score('K', 'R', buried=False)
        assert abs(score2) < 1.0


class TestEndToEndPipeline:
    """Test complete analysis pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_comprehensive_pipeline(self, mock_api_key, tmp_path):
        """Test full end-to-end analysis."""
        from alphagenome_plus.pipelines import ComprehensiveAnalysisPipeline
        
        pipeline = ComprehensiveAnalysisPipeline(
            api_key=mock_api_key,
            output_dir=str(tmp_path)
        )
        
        interval = genome.Interval(
            chromosome='chr22',
            start=35677410,
            end=36725986
        )
        
        variants = [
            genome.Variant(
                chromosome='chr22',
                position=36201698 + i * 1000,
                reference_bases='A',
                alternate_bases='T'
            )
            for i in range(5)
        ]
        
        with patch.object(pipeline.batch_processor, '_call_api', 
                         new_callable=AsyncMock) as mock_call:
            mock_call.return_value = Mock()
            
            results = await pipeline.run(interval, variants)
            
            assert 'predictions' in results
            assert 'scores' in results
            assert 'report' in results
            assert (tmp_path / "analysis_report.txt").exists()


# Fixtures
@pytest.fixture
def mock_api_key():
    return "test_api_key_12345"


@pytest.fixture
def sample_interval():
    return genome.Interval(
        chromosome='chr22',
        start=35677410,
        end=36725986
    )


@pytest.fixture
def sample_variants():
    return [
        genome.Variant(
            chromosome='chr22',
            position=36201698,
            reference_bases='A',
            alternate_bases='C'
        )
    ]
