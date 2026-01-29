"""Pathogenicity prediction using ensemble methods."""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from alphagenome.data import genome


@dataclass
class PathogenicityScore:
    """Pathogenicity prediction result."""
    score: float  # 0-1, higher = more pathogenic
    confidence: float
    category: str  # 'benign', 'likely_benign', 'uncertain', 'likely_pathogenic', 'pathogenic'
    contributing_factors: Dict[str, float]


class PathogenicityPredictor:
    """Ensemble pathogenicity predictor.
    
    Combines multiple features from AlphaGenome predictions with
    conservation scores and other genomic features to predict
    variant pathogenicity.
    
    Example:
        >>> predictor = PathogenicityPredictor()
        >>> predictor.train(training_variants, training_labels)
        >>> score = predictor.predict(
        ...     variant=test_variant,
        ...     alphagenome_features=features
        ... )
        >>> print(f"Pathogenicity: {score.category} (score: {score.score:.3f})")
    """
    
    def __init__(self, use_ensemble: bool = True):
        """Initialize pathogenicity predictor.
        
        Args:
            use_ensemble: Use ensemble of models
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Scikit-learn and XGBoost required. "
                "Install with: pip install 'alphagenome-plus[ml]'"
            )
        
        self.use_ensemble = use_ensemble
        self.models = self._initialize_models()
        self.is_trained = False
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize ensemble models.
        
        Returns:
            Dictionary of model instances
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            ),
        }
        return models
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Train pathogenicity models.
        
        Args:
            X: Feature matrix
            y: Labels (0=benign, 1=pathogenic)
            feature_names: Optional feature names
        """
        self.feature_names = feature_names
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y)
        
        self.is_trained = True
    
    def predict(
        self,
        variant: genome.Variant,
        alphagenome_features: Dict[str, np.ndarray],
        conservation_scores: Optional[Dict[str, float]] = None,
    ) -> PathogenicityScore:
        """Predict pathogenicity of variant.
        
        Args:
            variant: Genomic variant
            alphagenome_features: Features from AlphaGenome
            conservation_scores: Optional conservation scores
            
        Returns:
            PathogenicityScore
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Compile features
        features = self._compile_features(
            variant,
            alphagenome_features,
            conservation_scores,
        )
        
        # Get predictions from ensemble
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(features.reshape(1, -1))[0]
            prob = model.predict_proba(features.reshape(1, -1))[0, 1]
            predictions[name] = pred
            probabilities[name] = prob
        
        # Ensemble voting
        ensemble_score = np.mean(list(probabilities.values()))
        ensemble_confidence = 1.0 - np.std(list(probabilities.values()))
        
        # Categorize
        category = self._categorize_score(ensemble_score)
        
        # Feature importance
        importance = self._get_feature_importance()
        
        return PathogenicityScore(
            score=ensemble_score,
            confidence=ensemble_confidence,
            category=category,
            contributing_factors=importance,
        )
    
    def _compile_features(
        self,
        variant: genome.Variant,
        alphagenome_features: Dict[str, np.ndarray],
        conservation_scores: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Compile feature vector for prediction.
        
        Args:
            variant: Variant
            alphagenome_features: AlphaGenome features
            conservation_scores: Conservation scores
            
        Returns:
            Feature vector
        """
        features = []
        
        # AlphaGenome features
        if 'rna_seq_diff' in alphagenome_features:
            # Statistical features from difference
            diff = alphagenome_features['rna_seq_diff']
            features.extend([
                np.max(np.abs(diff)),
                np.mean(np.abs(diff)),
                np.std(diff),
            ])
        
        # Conservation scores
        if conservation_scores:
            features.extend([
                conservation_scores.get('phyloP', 0.0),
                conservation_scores.get('phastCons', 0.0),
                conservation_scores.get('GERP', 0.0),
            ])
        
        # Variant type features
        ref_len = len(variant.reference_bases)
        alt_len = len(variant.alternate_bases)
        features.extend([
            1 if ref_len == alt_len else 0,  # SNV
            1 if alt_len > ref_len else 0,   # Insertion
            1 if alt_len < ref_len else 0,   # Deletion
        ])
        
        return np.array(features)
    
    def _categorize_score(self, score: float) -> str:
        """Categorize pathogenicity score.
        
        Args:
            score: Pathogenicity score
            
        Returns:
            Category label
        """
        if score < 0.1:
            return 'benign'
        elif score < 0.3:
            return 'likely_benign'
        elif score < 0.7:
            return 'uncertain'
        elif score < 0.9:
            return 'likely_pathogenic'
        else:
            return 'pathogenic'
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from models.
        
        Returns:
            Dictionary of feature importances
        """
        # Average importance across models
        rf_importance = self.models['random_forest'].feature_importances_
        gb_importance = self.models['gradient_boosting'].feature_importances_
        xgb_importance = self.models['xgboost'].feature_importances_
        
        avg_importance = (rf_importance + gb_importance + xgb_importance) / 3
        
        if self.feature_names:
            return dict(zip(self.feature_names, avg_importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(avg_importance)}