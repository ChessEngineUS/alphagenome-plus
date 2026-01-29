# AlphaGenome-Plus Examples

This directory contains comprehensive examples demonstrating all features of the AlphaGenome-Plus toolkit.

## Overview

### 1. Quantum Variant Optimization (`01_quantum_variant_optimization.py`)

Demonstrates quantum-inspired optimization for variant prioritization:
- VQE-based variant ranking
- QAOA for clinical prioritization
- Feature encoding from AlphaGenome predictions
- Integration with ClinVar scores
- Multi-criteria clinical ranking

**Run:**
```bash
python examples/01_quantum_variant_optimization.py
```

**Requirements:**
- Qiskit >= 1.0
- qiskit-algorithms
- qiskit-optimization

### 2. Protein Folding Integration (`02_protein_folding_integration.py`)

Shows integration with AlphaFold3 and ESM-2:
- Structural impact prediction (ΔΔG, burial, pLDDT)
- Binding site analysis
- ESM-2 embedding differences
- Batch variant processing
- Combined genomic + structural scores

**Run:**
```bash
python examples/02_protein_folding_integration.py
```

**Requirements:**
- NumPy
- AlphaFold3 API key (optional)
- ESM-2 (optional, for production)

### 3. ML Training Pipeline (`03_ml_training_pipeline.py`)

Complete ML pipeline for variant effect prediction:
- PyTorch dataset creation
- Neural network architecture
- Training with validation
- Model evaluation and metrics
- Visualization of results
- Model saving/loading

**Run:**
```bash
python examples/03_ml_training_pipeline.py
```

**Requirements:**
- PyTorch >= 2.0
- Matplotlib

## Advanced Usage

### Combining All Features

For a complete analysis pipeline combining quantum optimization, protein folding, and ML:

```python
from alphagenome_plus.batch import BatchPredictor
from alphagenome_plus.quantum.vqe_optimizer import VariantVQEOptimizer
from alphagenome_plus.protein.alphafold_integration import AlphaFoldIntegrator
from alphagenome_plus.ml.training_pipeline import TrainingPipeline

# 1. Batch AlphaGenome predictions
predictor = BatchPredictor(api_key=API_KEY)
genomics_results = await predictor.predict_variants_async(variants)

# 2. Protein structural analysis
af_integrator = AlphaFoldIntegrator()
structural_impacts = af_integrator.batch_predict_structural_impacts(variants)

# 3. Quantum optimization for prioritization
vqe_optimizer = VariantVQEOptimizer()
rankings = vqe_optimizer.optimize_variant_prioritization(features)

# 4. ML-based scoring
ml_pipeline = TrainingPipeline(config)
ml_scores = ml_pipeline.predict(combined_features)
```

## Notebooks

Jupyter notebooks with interactive visualizations are available in the `notebooks/` directory:

- `tutorial_01_basic_usage.ipynb` - Getting started
- `tutorial_02_quantum_algorithms.ipynb` - Deep dive into VQE/QAOA
- `tutorial_03_structural_analysis.ipynb` - Protein folding integration
- `tutorial_04_ml_workflows.ipynb` - ML training and inference
- `case_study_clinvar.ipynb` - Real-world ClinVar analysis

## Data

Example data files are in `examples/data/`:
- `sample_variants.vcf` - Example VCF file
- `sample_proteins.fasta` - Protein sequences
- `clinvar_subset.tsv` - ClinVar annotations

## Citation

If you use these examples in your research, please cite:

```bibtex
@software{alphagenome_plus,
  author = {Your Name},
  title = {AlphaGenome-Plus: Enhanced toolkit for genomic variant analysis},
  year = {2026},
  url = {https://github.com/ChessEngineUS/alphagenome-plus}
}
```
