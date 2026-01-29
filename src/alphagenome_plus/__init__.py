"""AlphaGenome-Plus: Enhanced genomic analysis toolkit.

This package extends the AlphaGenome API with advanced features including:
- Batch processing and intelligent caching
- ML integration with PyTorch/TensorFlow
- Quantum-inspired optimization algorithms
- Protein folding integration
- Advanced analysis pipelines
"""

__version__ = "0.1.0"
__author__ = "Tommaso R. Marena"

from alphagenome_plus.batch import BatchPredictor
from alphagenome_plus.cache import PredictionCache

__all__ = [
    "BatchPredictor",
    "PredictionCache",
    "__version__",
]