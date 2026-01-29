"""Machine learning integration module."""

from alphagenome_plus.ml.embeddings import FeatureExtractor, EmbeddingModel
from alphagenome_plus.ml.fine_tuning import FineTuningAdapter
from alphagenome_plus.ml.pathogenicity import PathogenicityPredictor

__all__ = [
    "FeatureExtractor",
    "EmbeddingModel",
    "FineTuningAdapter",
    "PathogenicityPredictor",
]