# AlphaGenome-Plus API Reference

## Quantum Optimization

### `VariantVQEOptimizer`

Variational Quantum Eigensolver for variant prioritization.

#### Methods

##### `__init__(num_qubits, num_layers, optimizer, max_iterations)`

Initialize VQE optimizer.

**Parameters:**
- `num_qubits` (int): Number of qubits (default: 4)
- `num_layers` (int): Circuit depth (default: 3)
- `optimizer` (str): Classical optimizer ('SPSA', 'COBYLA', 'SLSQP')
- `max_iterations` (int): Maximum optimization iterations

##### `optimize_variant_prioritization(variant_features, importance_weights)`

Optimize variant prioritization.

**Parameters:**
- `variant_features` (List[np.ndarray]): Feature vectors
- `importance_weights` (Optional[np.ndarray]): Hamiltonian weights

**Returns:**
- Dict with keys: `optimal_energy`, `optimal_params`, `variant_rankings`

---

### `VariantPrioritizationQAOA`

QAOA for clinical variant prioritization.

#### Methods

##### `solve_prioritization(pathogenicity_scores, interaction_matrix)`

Solve variant selection problem.

**Parameters:**
- `pathogenicity_scores` (np.ndarray): Scores for each variant
- `interaction_matrix` (Optional[np.ndarray]): Variant interactions

**Returns:**
- Dict with selected variant indices and objective value

---

## Protein Folding

### `AlphaFoldIntegrator`

Integration with AlphaFold3 for structural analysis.

#### Methods

##### `predict_variant_structural_impact(...)`

Predict structural impact of coding variant.

**Parameters:**
- `variant_id` (str): Variant identifier
- `protein_sequence` (str): Full protein sequence
- `variant_position` (int): Position in protein (1-indexed)
- `reference_aa` (str): Reference amino acid
- `alternate_aa` (str): Alternate amino acid
- `alphagenome_scores` (Optional[Dict]): AlphaGenome predictions

**Returns:**
- `VariantStructuralImpact` object

##### `batch_predict_structural_impacts(variants)`

Batch prediction for multiple variants.

**Parameters:**
- `variants` (List[Tuple]): List of (id, seq, pos, ref, alt)

**Returns:**
- List of `VariantStructuralImpact` objects

---

### `ESMEmbeddingAnalyzer`

ESM-2 protein language model analysis.

#### Methods

##### `compute_embedding_difference(wt_sequence, mut_sequence)`

Compute embedding difference between wildtype and mutant.

**Returns:**
- np.ndarray of shape (1280,)

##### `predict_functional_impact(embedding_diff)`

Predict functional impact from embedding difference.

**Returns:**
- float: Impact score (0-1)

---

## Machine Learning

### `TrainingPipeline`

Complete ML training pipeline.

#### Methods

##### `build_model(input_dim, output_dims)`

Build model architecture.

**Parameters:**
- `input_dim` (int): Number of input features
- `output_dims` (Dict[str, int]): Output head dimensions

##### `train(train_dataset, val_dataset, loss_weights)`

Train the model.

**Parameters:**
- `train_dataset` (VariantEffectDataset): Training data
- `val_dataset` (Optional): Validation data
- `loss_weights` (Optional[Dict]): Weights per output head

**Returns:**
- Training history dictionary

##### `predict(features)`

Make predictions.

**Parameters:**
- `features` (np.ndarray): Input features [N, input_dim]

**Returns:**
- Dict[str, np.ndarray]: Predictions per output head

##### `save_model(path)` / `load_model(path, input_dim, output_dims)`

Save and load trained models.

---

## Data Structures

### `VariantStructuralImpact`

Structural impact assessment.

**Attributes:**
- `variant_id` (str)
- `protein_id` (str)
- `position` (int)
- `reference_aa` (str)
- `alternate_aa` (str)
- `ddg_stability` (float): ΔΔG in kcal/mol
- `burial_score` (float): 0-1
- `secondary_structure` (str): 'helix', 'sheet', or 'loop'
- `binding_site_affected` (bool)
- `plddt_change` (float)
- `interface_disruption` (float): 0-1
- `structural_pathogenicity` (float): 0-1

---

### `TrainingConfig`

Training configuration.

**Attributes:**
- `batch_size` (int): Default 32
- `learning_rate` (float): Default 1e-4
- `num_epochs` (int): Default 50
- `weight_decay` (float): Default 1e-5
- `dropout_rate` (float): Default 0.2
- `hidden_dims` (List[int]): Hidden layer sizes
- `device` (str): 'cuda' or 'cpu'
