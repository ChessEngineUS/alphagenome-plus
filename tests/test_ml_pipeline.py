"""Tests for ML training pipeline."""

import pytest
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alphagenome_plus.ml.training_pipeline import (
    TrainingConfig,
    VariantEffectDataset,
    VariantEffectPredictor,
    TrainingPipeline
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTrainingConfig:
    
    def test_default_config(self):
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 50
        assert config.hidden_dims is not None
    
    def test_custom_config(self):
        config = TrainingConfig(
            batch_size=64,
            learning_rate=1e-3,
            hidden_dims=[256, 128]
        )
        
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.hidden_dims == [256, 128]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestVariantEffectDataset:
    
    def test_dataset_creation(self):
        features = np.random.randn(100, 5)
        labels = np.random.randn(100)
        
        dataset = VariantEffectDataset(features, labels)
        
        assert len(dataset) == 100
    
    def test_dataset_indexing(self):
        features = np.random.randn(10, 5)
        labels = np.random.randn(10)
        
        dataset = VariantEffectDataset(features, labels, normalize=False)
        
        x, y = dataset[0]
        assert x.shape == (5,)
        assert isinstance(y, torch.Tensor)
    
    def test_normalization(self):
        features = np.random.randn(100, 5) * 10 + 5
        labels = np.random.randn(100)
        
        dataset = VariantEffectDataset(features, labels, normalize=True)
        
        # Features should be normalized
        all_features = torch.stack([dataset[i][0] for i in range(len(dataset))])
        assert torch.abs(all_features.mean(0)).max() < 0.1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestVariantEffectPredictor:
    
    def test_model_creation(self):
        model = VariantEffectPredictor(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dims={'pathogenicity': 1, 'splicing': 1},
            dropout_rate=0.2
        )
        
        assert len(model.heads) == 2
    
    def test_forward_pass(self):
        model = VariantEffectPredictor(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dims={'score': 1}
        )
        
        x = torch.randn(10, 5)  # Batch of 10
        outputs = model(x)
        
        assert 'score' in outputs
        assert outputs['score'].shape == (10, 1)
    
    def test_multiple_heads(self):
        model = VariantEffectPredictor(
            input_dim=5,
            hidden_dims=[32],
            output_dims={'path': 1, 'splice': 2, 'expr': 3}
        )
        
        x = torch.randn(5, 5)
        outputs = model(x)
        
        assert len(outputs) == 3
        assert outputs['path'].shape == (5, 1)
        assert outputs['splice'].shape == (5, 2)
        assert outputs['expr'].shape == (5, 3)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTrainingPipeline:
    
    def test_pipeline_initialization(self):
        config = TrainingConfig(num_epochs=5)
        pipeline = TrainingPipeline(config)
        
        assert pipeline.config.num_epochs == 5
        assert pipeline.model is None
    
    def test_model_building(self):
        config = TrainingConfig()
        pipeline = TrainingPipeline(config)
        
        pipeline.build_model(
            input_dim=5,
            output_dims={'score': 1}
        )
        
        assert pipeline.model is not None
        assert pipeline.optimizer is not None
    
    def test_training_workflow(self):
        # Small dataset for quick test
        features = np.random.randn(50, 5)
        labels = np.random.randn(50)
        
        train_dataset = VariantEffectDataset(features[:40], labels[:40])
        val_dataset = VariantEffectDataset(features[40:], labels[40:])
        
        config = TrainingConfig(
            batch_size=10,
            num_epochs=2,
            hidden_dims=[16]
        )
        pipeline = TrainingPipeline(config)
        pipeline.build_model(input_dim=5, output_dims={'score': 1})
        
        history = pipeline.train(train_dataset, val_dataset)
        
        assert len(history) == 2
        assert 'train_loss' in history[0]
        assert 'val_loss' in history[0]
    
    def test_prediction(self):
        features = np.random.randn(20, 5)
        labels = np.random.randn(20)
        
        dataset = VariantEffectDataset(features, labels)
        
        config = TrainingConfig(num_epochs=1, hidden_dims=[16])
        pipeline = TrainingPipeline(config)
        pipeline.build_model(input_dim=5, output_dims={'score': 1})
        pipeline.train(dataset)
        
        # Make predictions
        test_features = np.random.randn(5, 5)
        predictions = pipeline.predict(test_features)
        
        assert 'score' in predictions
        assert predictions['score'].shape == (5, 1)
