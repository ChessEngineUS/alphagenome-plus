"""Example: ML integration for variant effect prediction.

Demonstrates how to train a deep learning model on AlphaGenome embeddings
for downstream variant classification tasks.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from alphagenome_plus.ml.embeddings import (
    AlphaGenomeEmbeddingExtractor,
    VariantEffectPredictor,
    EmbeddingConfig
)
from alphagenome_plus.ml.training import train_variant_classifier


def main():
    # Configuration
    embedding_config = EmbeddingConfig(
        embedding_dim=512,
        pooling_strategy='mean',
        normalize=True,
        device='cuda'
    )
    
    # Initialize embedding extractor
    extractor = AlphaGenomeEmbeddingExtractor(embedding_config)
    
    # Load pre-computed AlphaGenome predictions
    # (In practice, these would come from batch processing)
    print("Loading AlphaGenome predictions...")
    
    # Simulate predictions for demonstration
    n_samples = 1000
    predictions_list = []
    labels = []
    
    for i in range(n_samples):
        # Simulated prediction outputs
        pred = {
            'rna_seq': np.random.randn(100, 32),
            'chip_seq': np.random.randn(100, 16),
            'cage': np.random.randn(100, 8)
        }
        predictions_list.append(pred)
        
        # Simulated labels (0: benign, 1: likely benign, 2: pathogenic)
        labels.append(np.random.randint(0, 3))
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    for pred in predictions_list:
        emb = extractor.extract_from_predictions(pred, sequence_length=131072)
        embeddings.append(emb.cpu().numpy())
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Split train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = embeddings[:split_idx], embeddings[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = VariantEffectPredictor(
        embedding_dim=512,
        hidden_dims=[256, 128, 64],
        num_classes=3,
        dropout=0.3
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Train model
    print("\nTraining variant classifier...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * correct / total
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        print(f"Epoch {epoch+1}/10: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'variant_classifier.pth')
    print("\nModel saved to variant_classifier.pth")


if __name__ == "__main__":
    main()
