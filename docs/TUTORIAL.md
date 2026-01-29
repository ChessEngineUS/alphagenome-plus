# AlphaGenome-Plus Tutorial

Comprehensive guide to using AlphaGenome-Plus for genomic variant analysis.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Batch Processing](#batch-processing)
4. [ML Integration](#ml-integration)
5. [Quantum Optimization](#quantum-optimization)
6. [Protein Structure Analysis](#protein-structure-analysis)
7. [Advanced Pipelines](#advanced-pipelines)

## Installation

### Basic Installation

```bash
# Clone repository
git clone https://github.com/ChessEngineUS/alphagenome-plus.git
cd alphagenome-plus

# Install base package
pip install -e .
```

### Optional Dependencies

```bash
# For ML features
pip install -e ".[ml]"

# For quantum optimization
pip install -e ".[quantum]"

# For protein structure analysis
pip install -e ".[protein]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

### 1. Basic Variant Scoring

```python
import asyncio
from alphagenome.data import genome
from alphagenome_plus.batch import BatchVariantProcessor

async def score_variants():
    processor = BatchVariantProcessor(api_key="YOUR_API_KEY")
    
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
    
    result = await processor.process_single_variant(
        interval=interval,
        variant=variant,
        ontology_terms=['UBERON:0001157']
    )
    
    return result

result = asyncio.run(score_variants())
```

### 2. Enable Caching

```python
from alphagenome_plus.cache import PredictionCache

cache = PredictionCache(cache_dir="./variant_cache")
processor = BatchVariantProcessor(
    api_key="YOUR_API_KEY",
    cache=cache
)

# Subsequent identical requests will use cached results
```

## Batch Processing

### High-Throughput Analysis

```python
from alphagenome_plus.batch import BatchConfig

config = BatchConfig(
    max_concurrent=10,        # Process 10 variants in parallel
    retry_attempts=3,         # Retry failed requests
    rate_limit_per_minute=100,# Respect API limits
    use_cache=True
)

processor = BatchVariantProcessor(
    api_key="YOUR_API_KEY",
    config=config,
    cache=cache
)

# Process hundreds of variants efficiently
variants = [...]  # List of genome.Variant objects
results = await processor.process_variants(
    interval=interval,
    variants=variants,
    show_progress=True
)
```

### VCF File Processing

```python
from alphagenome_plus.batch import VCFProcessor

vcf_processor = VCFProcessor(api_key="YOUR_API_KEY")
results = await vcf_processor.process_vcf(
    vcf_path="variants.vcf",
    output_path="results.tsv",
    ontology_terms=['UBERON:0001157']
)
```

## ML Integration

### Extract Embeddings

```python
from alphagenome_plus.ml.embeddings import (
    AlphaGenomeEmbeddingExtractor,
    EmbeddingConfig
)

config = EmbeddingConfig(
    embedding_dim=512,
    pooling_strategy='mean',
    normalize=True
)

extractor = AlphaGenomeEmbeddingExtractor(config)

# Extract from AlphaGenome predictions
embed = extractor.extract_from_predictions(
    predictions=result,
    sequence_length=131072
)
```

### Train Variant Classifier

```python
import torch
from alphagenome_plus.ml.embeddings import VariantEffectPredictor

model = VariantEffectPredictor(
    embedding_dim=512,
    hidden_dims=[256, 128, 64],
    num_classes=3,  # benign, uncertain, pathogenic
    dropout=0.3
)

# Train on your labeled data
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

### Contrastive Learning

```python
from alphagenome_plus.ml.embeddings import (
    ContrastiveLearningHead,
    GenomicContrastiveLoss
)

# Project embeddings to contrastive space
head = ContrastiveLearningHead(input_dim=512, projection_dim=256)
loss_fn = GenomicContrastiveLoss(temperature=0.07)

# Train with augmented pairs
z_i = head(embeddings_view1)
z_j = head(embeddings_view2)
loss = loss_fn(z_i, z_j)
```

## Quantum Optimization

### Variant Prioritization with QAOA

```python
import numpy as np
from alphagenome_plus.quantum.qaoa_optimizer import (
    VariantPrioritizationQAOA,
    QAOAConfig
)

# Define variant scores and correlations
scores = np.array([...])  # Pathogenicity scores
correlations = np.array([...])  # Variant correlation matrix

# Configure QAOA
config = QAOAConfig(
    num_layers=3,
    max_iterations=100,
    optimizer='COBYLA',
    shots=1024
)

qaoa = VariantPrioritizationQAOA(config)

# Select top k variants minimizing redundancy
selected_indices, objective = qaoa.optimize(
    scores=scores,
    correlations=correlations,
    k=10  # Select 10 variants
)

print(f"Selected variants: {selected_indices}")
print(f"Objective value: {objective:.4f}")
```

### Quantum Feature Selection

```python
from alphagenome_plus.quantum.qaoa_optimizer import QuantumFeatureSelector

selector = QuantumFeatureSelector(n_features=100)
selected_features = selector.select_features(
    X=feature_matrix,
    y=labels,
    k=20  # Select 20 most informative features
)
```

## Protein Structure Analysis

### AlphaFold Integration

```python
from alphagenome_plus.protein.alphafold_integration import (
    AlphaFoldStructureAnalyzer
)

analyzer = AlphaFoldStructureAnalyzer()

# Analyze missense variant
impact = analyzer.analyze_variant_impact(
    uniprot_id="P04637",  # TP53
    position=175,
    ref_aa="R",
    alt_aa="H"
)

print(f"Stability change: {impact.stability_score:.2f} kcal/mol")
print(f"Buried: {impact.buried}")
print(f"Contacts disrupted: {impact.contacts_disrupted}")
```

### ESM-2 Embeddings

```python
from alphagenome_plus.protein.alphafold_integration import ESMEmbeddingAnalyzer

esm = ESMEmbeddingAnalyzer()
esm.load_model()

score = esm.compute_mutation_score(
    sequence="MEEPQSDPSVEP...",
    position=175,
    alt_aa="H"
)

print(f"Mutation effect score: {score:.4f}")
```

## Advanced Pipelines

### Complete Analysis Pipeline

```python
from alphagenome_plus.pipelines import ComprehensiveAnalysisPipeline

pipeline = ComprehensiveAnalysisPipeline(
    api_key="YOUR_API_KEY",
    enable_ml=True,
    enable_quantum=True,
    enable_structure=True
)

results = await pipeline.analyze(
    interval=interval,
    variants=variants,
    output_dir="./analysis_results"
)

# Results include:
# - AlphaGenome predictions
# - ML-based classifications
# - Prioritized variant list
# - Structural impacts
# - Clinical interpretations
```

### Custom Pipeline

```python
from alphagenome_plus.pipelines import Pipeline, PipelineStep

class MyCustomStep(PipelineStep):
    async def execute(self, data):
        # Your custom analysis
        return processed_data

pipeline = Pipeline([
    BatchPredictionStep(api_key="YOUR_API_KEY"),
    EmbeddingExtractionStep(),
    MyCustomStep(),
    ReportGenerationStep()
])

results = await pipeline.run(variants)
```

## Performance Optimization

### GPU Acceleration

```python
from alphagenome_plus.acceleration import GPUAccelerator

gpu = GPUAccelerator(device='cuda')
processor = BatchVariantProcessor(
    api_key="YOUR_API_KEY",
    accelerator=gpu
)

# Post-processing runs on GPU
```

### Distributed Processing

```python
from alphagenome_plus.batch import DistributedProcessor

processor = DistributedProcessor(
    api_key="YOUR_API_KEY",
    num_workers=4,
    backend='ray'  # or 'dask'
)

results = await processor.process_variants(variants)
```

## Best Practices

### 1. Use Caching Aggressively

```python
# Enable persistent cache
cache = PredictionCache(
    cache_dir="./cache",
    max_size_gb=10
)
```

### 2. Batch Similar Requests

```python
# Group variants by interval for efficiency
from alphagenome_plus.utils import group_variants_by_interval

grouped = group_variants_by_interval(variants, window_size=1_000_000)
for interval, interval_variants in grouped.items():
    results = await processor.process_variants(interval, interval_variants)
```

### 3. Monitor Progress

```python
results = await processor.process_variants(
    variants=variants,
    show_progress=True,
    progress_callback=lambda n, total: print(f"{n}/{total}")
)
```

### 4. Handle Errors Gracefully

```python
config = BatchConfig(
    retry_attempts=3,
    timeout_seconds=300,
    continue_on_error=True
)

results = await processor.process_variants(
    variants=variants,
    config=config
)

# Check for failures
failed = [v for v, r in results.items() if r is None]
```

## Troubleshooting

### API Rate Limits

```python
config = BatchConfig(
    rate_limit_per_minute=50,  # Reduce if hitting limits
    backoff_factor=2.0         # Exponential backoff
)
```

### Memory Issues

```python
# Process in chunks
chunk_size = 100
for i in range(0, len(variants), chunk_size):
    chunk = variants[i:i+chunk_size]
    results = await processor.process_variants(chunk)
    # Save results incrementally
```

### CUDA Out of Memory

```python
# Reduce batch size
config = EmbeddingConfig(
    device='cuda',
    batch_size=16  # Reduce from default 32
)
```

## Citation

If you use AlphaGenome-Plus in your research, please cite:

```bibtex
@software{alphagenome_plus_2026,
  title={AlphaGenome-Plus: Enhanced Toolkit for Genomic Variant Analysis},
  author={Your Name},
  year={2026},
  url={https://github.com/ChessEngineUS/alphagenome-plus}
}
```

And the original AlphaGenome paper:

```bibtex
@article{alphagenome2025,
  title={AlphaGenome: advancing regulatory variant effect prediction},
  author={Avsec et al.},
  year={2025},
  doi={https://doi.org/10.1101/2025.06.25.661532}
}
```
