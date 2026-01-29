"""Unit tests for ML integration modules."""

import pytest
import numpy as np
import torch
import jax.numpy as jnp
from unittest.mock import Mock, MagicMock

from alphagenome_plus.ml.embedding_extractor import (
    EmbeddingConfig,
    AlphaGenomeEmbeddingExtractor,
    PyTorchEmbeddingAdapter,
    JAXEmbeddingAdapter,
    FineTuningDataset,
    create_downstream_classifier
)


class TestEmbeddingConfig:
    """Test EmbeddingConfig."""
    
    def test_default_config(self):
        config = EmbeddingConfig()
        assert config.embedding_dim == 768
        assert config.pooling_strategy == 'mean'
        assert config.normalize is True
    
    def test_custom_config(self):
        config = EmbeddingConfig(
            embedding_dim=512,
            pooling_strategy='max',
            normalize=False
        )
        assert config.embedding_dim == 512
        assert config.pooling_strategy == 'max'
        assert config.normalize is False


class TestAlphaGenomeEmbeddingExtractor:
    """Test embedding extraction."""
    
    @pytest.fixture
    def mock_model(self):
        return Mock()
    
    @pytest.fixture
    def extractor(self, mock_model):
        config = EmbeddingConfig(embedding_dim=768)
        return AlphaGenomeEmbeddingExtractor(mock_model, config)
    
    def test_pool_sequence_mean(self, extractor):
        seq = np.random.randn(10, 64)
        pooled = extractor._pool_sequence(seq, strategy='mean')
        
        assert pooled.shape == (64,)
        assert np.allclose(pooled, np.mean(seq, axis=0))
    
    def test_pool_sequence_max(self, extractor):
        seq = np.random.randn(10, 64)
        pooled = extractor._pool_sequence(seq, strategy='max')
        
        assert pooled.shape == (64,)
        assert np.allclose(pooled, np.max(seq, axis=0))
    
    def test_pool_sequence_cls(self, extractor):
        seq = np.random.randn(10, 64)
        pooled = extractor._pool_sequence(seq, strategy='cls')
        
        assert pooled.shape == (64,)
        assert np.allclose(pooled, seq[0])
    
    def test_pool_sequence_invalid_strategy(self, extractor):
        seq = np.random.randn(10, 64)
        
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            extractor._pool_sequence(seq, strategy='invalid')
    
    def test_extract_from_prediction(self, extractor):
        # Mock prediction output
        mock_output = Mock()
        mock_output.rna_seq = Mock(data=np.random.randn(100, 384))
        mock_output.chromatin_features = Mock(data=np.random.randn(100, 384))
        
        embedding = extractor.extract_from_prediction(mock_output)
        
        # Should concatenate and normalize
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)  # 384 + 384
        
        # Check normalization
        if extractor.config.normalize:
            assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


class TestPyTorchEmbeddingAdapter:
    """Test PyTorch adapter."""
    
    @pytest.fixture
    def mock_extractor(self):
        extractor = Mock()
        extractor.config = EmbeddingConfig(embedding_dim=768)
        extractor.extract_from_prediction = Mock(
            return_value=np.random.randn(768)
        )
        return extractor
    
    @pytest.fixture
    def adapter(self, mock_extractor):
        return PyTorchEmbeddingAdapter(mock_extractor, projection_dim=256)
    
    def test_initialization(self, mock_extractor):
        adapter = PyTorchEmbeddingAdapter(mock_extractor)
        assert adapter.projection is None
        
        adapter_with_proj = PyTorchEmbeddingAdapter(
            mock_extractor, projection_dim=256
        )
        assert adapter_with_proj.projection is not None
    
    def test_forward_without_projection(self, mock_extractor):
        adapter = PyTorchEmbeddingAdapter(mock_extractor, projection_dim=None)
        
        mock_outputs = [Mock(), Mock()]
        result = adapter.forward(mock_outputs)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 768)
    
    def test_forward_with_projection(self, adapter, mock_extractor):
        mock_outputs = [Mock(), Mock()]
        result = adapter.forward(mock_outputs)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 256)  # Projected dimension


class TestJAXEmbeddingAdapter:
    """Test JAX adapter."""
    
    @pytest.fixture
    def mock_extractor(self):
        extractor = Mock()
        extractor.extract_from_prediction = Mock(
            return_value=np.random.randn(768)
        )
        return extractor
    
    @pytest.fixture
    def adapter(self, mock_extractor):
        return JAXEmbeddingAdapter(mock_extractor)
    
    def test_extract_jax(self, adapter, mock_extractor):
        mock_outputs = [Mock(), Mock()]
        result = adapter.extract_jax(mock_outputs)
        
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (2, 768)
    
    def test_apply_projection(self, adapter):
        embeddings = jnp.array(np.random.randn(4, 768))
        
        params = {
            'weight': jnp.array(np.random.randn(768, 256)),
            'bias': jnp.array(np.random.randn(256)),
            'ln_scale': jnp.ones(256),
            'ln_bias': jnp.zeros(256)
        }
        
        projected = adapter.apply_projection(embeddings, params)
        
        assert projected.shape == (4, 256)
        # ReLU ensures non-negative
        assert jnp.all(projected >= 0)


class TestFineTuningDataset:
    """Test PyTorch dataset."""
    
    @pytest.fixture
    def mock_extractor(self):
        extractor = Mock()
        extractor.extract_batch = Mock(
            return_value=np.random.randn(5, 768)
        )
        return extractor
    
    @pytest.fixture
    def dataset(self, mock_extractor):
        intervals = [Mock() for _ in range(5)]
        labels = [0, 1, 0, 1, 1]
        return FineTuningDataset(intervals, labels, mock_extractor)
    
    def test_length(self, dataset):
        assert len(dataset) == 5
    
    def test_getitem(self, dataset):
        embed, label = dataset[0]
        
        assert isinstance(embed, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert embed.shape == (768,)
        assert label.shape == ()
    
    def test_getitem_with_transform(self, mock_extractor):
        transform = lambda x: x * 2
        dataset = FineTuningDataset(
            [Mock() for _ in range(5)],
            [0, 1, 0, 1, 1],
            mock_extractor,
            transform=transform
        )
        
        embed, _ = dataset[0]
        # Transform should have been applied
        assert isinstance(embed, torch.Tensor)


class TestCreateDownstreamClassifier:
    """Test classifier creation."""
    
    def test_creates_valid_model(self):
        model = create_downstream_classifier(
            input_dim=768,
            num_classes=3,
            hidden_dims=[512, 256]
        )
        
        assert isinstance(model, torch.nn.Sequential)
        
        # Test forward pass
        x = torch.randn(4, 768)
        output = model(x)
        
        assert output.shape == (4, 3)
    
    def test_single_hidden_layer(self):
        model = create_downstream_classifier(
            input_dim=512,
            num_classes=2,
            hidden_dims=[256]
        )
        
        x = torch.randn(2, 512)
        output = model(x)
        
        assert output.shape == (2, 2)
