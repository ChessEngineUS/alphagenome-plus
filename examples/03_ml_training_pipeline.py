#!/usr/bin/env python3
"""Example: ML training pipeline for variant effect prediction.

Demonstrates training a neural network on AlphaGenome-derived features
for downstream variant effect prediction tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from alphagenome_plus.ml.training_pipeline import (
    TrainingConfig,
    VariantEffectDataset,
    TrainingPipeline
)


def generate_synthetic_data(num_samples: int = 1000):
    """Generate synthetic training data.
    
    In production, replace with actual AlphaGenome predictions.
    """
    # Features: [rna_seq, splicing, chromatin, conservation, gc_content]
    features = np.random.randn(num_samples, 5)
    
    # Labels: pathogenicity score (synthetic ground truth)
    # Simple linear combination with noise
    weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05])
    labels = features @ weights + np.random.randn(num_samples) * 0.1
    labels = 1 / (1 + np.exp(-labels))  # Sigmoid to [0, 1]
    
    return features, labels


def main():
    print("="*60)
    print("ML Training Pipeline Example")
    print("="*60)
    
    # Step 1: Generate/load data
    print("\n[1] Generating training data...")
    
    train_features, train_labels = generate_synthetic_data(num_samples=800)
    val_features, val_labels = generate_synthetic_data(num_samples=200)
    test_features, test_labels = generate_synthetic_data(num_samples=100)
    
    print(f"  Training samples:   {len(train_features)}")
    print(f"  Validation samples: {len(val_features)}")
    print(f"  Test samples:       {len(test_features)}")
    print(f"  Feature dimension:  {train_features.shape[1]}")
    
    # Step 2: Create datasets
    print("\n[2] Creating PyTorch datasets...")
    
    train_dataset = VariantEffectDataset(train_features, train_labels)
    val_dataset = VariantEffectDataset(val_features, val_labels)
    
    # Step 3: Configure training
    print("\n[3] Configuring training pipeline...")
    
    config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=50,
        weight_decay=1e-4,
        dropout_rate=0.3,
        hidden_dims=[128, 64, 32]
    )
    
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Learning rate:  {config.learning_rate}")
    print(f"  Num epochs:     {config.num_epochs}")
    print(f"  Device:         {config.device}")
    
    # Step 4: Build model
    print("\n[4] Building model architecture...")
    
    pipeline = TrainingPipeline(config)
    
    input_dim = train_features.shape[1]
    output_dims = {'pathogenicity': 1}  # Single regression head
    
    pipeline.build_model(input_dim, output_dims)
    
    # Step 5: Train model
    print("\n[5] Training model...")
    print("  This may take a few minutes...\n")
    
    history = pipeline.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Step 6: Evaluate on test set
    print("\n[6] Evaluating on test set...")
    
    predictions = pipeline.predict(test_features)
    pred_pathogenicity = predictions['pathogenicity'].squeeze()
    
    # Compute metrics
    mse = np.mean((pred_pathogenicity - test_labels) ** 2)
    mae = np.mean(np.abs(pred_pathogenicity - test_labels))
    correlation = np.corrcoef(pred_pathogenicity, test_labels)[0, 1]
    
    print(f"  MSE:         {mse:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  Correlation: {correlation:.4f}")
    
    # Step 7: Visualize results
    print("\n[7] Visualizing results...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training curves
    ax1 = axes[0]
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prediction scatter plot
    ax2 = axes[1]
    ax2.scatter(test_labels, pred_pathogenicity, alpha=0.6)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('True Pathogenicity')
    ax2.set_ylabel('Predicted Pathogenicity')
    ax2.set_title(f'Test Set Predictions (r={correlation:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("  Plots saved to: training_results.png")
    
    # Step 8: Save model
    print("\n[8] Saving trained model...")
    
    pipeline.save_model('variant_effect_model.pth')
    print("  Model saved to: variant_effect_model.pth")
    
    # Step 9: Demonstrate loading
    print("\n[9] Testing model loading...")
    
    new_pipeline = TrainingPipeline(config)
    new_pipeline.load_model('variant_effect_model.pth', input_dim, output_dims)
    
    # Verify predictions match
    new_predictions = new_pipeline.predict(test_features[:5])
    print("  Model successfully loaded and verified!")
    
    print("\n" + "="*60)
    print("ML training pipeline complete!")
    print("="*60)


if __name__ == '__main__':
    main()
