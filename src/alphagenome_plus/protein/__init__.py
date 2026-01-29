"""Protein folding and structural analysis integration."""

from alphagenome_plus.protein.analyzer import ProteinAnalyzer
from alphagenome_plus.protein.esm_integration import ESMEmbeddings

__all__ = [
    "ProteinAnalyzer",
    "ESMEmbeddings",
]