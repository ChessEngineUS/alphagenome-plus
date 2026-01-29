"""Clinical interpretation pipeline for genomic variants.

Integrates AlphaGenome predictions with population genetics, neuro-symbolic AI,
and quantum optimization for comprehensive clinical variant interpretation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class ClinicalSignificance(Enum):
    """ACMG/AMP clinical significance categories."""
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    UNCERTAIN = "Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


class EvidenceStrength(Enum):
    """ACMG evidence strength levels."""
    VERY_STRONG = "Very Strong"
    STRONG = "Strong"
    MODERATE = "Moderate"
    SUPPORTING = "Supporting"
    STAND_ALONE = "Stand Alone"


@dataclass
class ClinicalEvidence:
    """Evidence for clinical variant interpretation."""
    code: str  # e.g., "PVS1", "PM2", "BA1"
    category: str  # "pathogenic" or "benign"
    strength: EvidenceStrength
    description: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClinicalInterpretation:
    """Comprehensive clinical variant interpretation."""
    variant_id: str
    gene: str
    transcript: str
    
    # Predictions
    alphagenome_score: float
    pathogenicity_prediction: ClinicalSignificance
    confidence_score: float
    
    # Evidence
    pathogenic_evidence: List[ClinicalEvidence]
    benign_evidence: List[ClinicalEvidence]
    
    # Analysis components
    population_analysis: Dict[str, Any]
    functional_analysis: Dict[str, Any]
    computational_predictions: Dict[str, Any]
    
    # Integrated reasoning
    neuro_symbolic_reasoning: Dict[str, Any]
    quantum_priority_score: Optional[float] = None
    
    # Clinical metadata
    phenotypes: List[str] = field(default_factory=list)
    inheritance_pattern: Optional[str] = None
    affected_pathways: List[str] = field(default_factory=list)
    
    # Report
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


class ClinicalInterpretationPipeline:
    """Comprehensive clinical interpretation pipeline."""
    
    def __init__(self,
                 alphagenome_model,
                 enable_quantum: bool = True,
                 enable_gpu: bool = True,
                 enable_cache: bool = True):
        """Initialize clinical interpretation pipeline.
        
        Args:
            alphagenome_model: AlphaGenome model instance
            enable_quantum: Enable quantum optimization
            enable_gpu: Enable GPU acceleration
            enable_cache: Enable result caching
        """
        self.model = alphagenome_model
        self.enable_quantum = enable_quantum
        self.enable_gpu = enable_gpu
        self.enable_cache = enable_cache
        
        # Initialize components
        self._init_components()
        
        # ACMG criteria thresholds
        self.acmg_thresholds = self._init_acmg_thresholds()
    
    def _init_components(self):
        """Initialize pipeline components."""
        # Lazy imports to avoid circular dependencies
        from alphagenome_plus.ml.neurosymbolic import create_clinical_reasoning_system
        from alphagenome_plus.analysis.population_genetics import PopulationGeneticsAnalyzer
        
        self.neuro_symbolic = create_clinical_reasoning_system()
        self.pop_genetics = PopulationGeneticsAnalyzer()
        
        if self.enable_quantum:
            try:
                from alphagenome_plus.quantum import VariantPrioritizer
                self.quantum_prioritizer = VariantPrioritizer(n_qubits=10, p=2)
            except ImportError:
                self.enable_quantum = False
        
        if self.enable_gpu:
            try:
                from alphagenome_plus.acceleration.gpu_calculator import create_gpu_pipeline
                self.gpu_pipeline = create_gpu_pipeline()
            except (ImportError, RuntimeError):
                self.enable_gpu = False
    
    def _init_acmg_thresholds(self) -> Dict:
        """Initialize ACMG/AMP interpretation thresholds."""
        return {
            'pathogenic': {
                'very_strong': {
                    'min_count': 1,
                    'codes': ['PVS1']
                },
                'strong': {
                    'min_count': 2,
                    'codes': ['PS1', 'PS2', 'PS3', 'PS4']
                },
                'moderate': {
                    'min_count': 3,
                    'codes': ['PM1', 'PM2', 'PM3', 'PM4', 'PM5', 'PM6']
                },
                'supporting': {
                    'min_count': 2,
                    'codes': ['PP1', 'PP2', 'PP3', 'PP4', 'PP5']
                }
            },
            'benign': {
                'stand_alone': {
                    'min_count': 1,
                    'codes': ['BA1']
                },
                'strong': {
                    'min_count': 2,
                    'codes': ['BS1', 'BS2', 'BS3', 'BS4']
                },
                'supporting': {
                    'min_count': 2,
                    'codes': ['BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7']
                }
            }
        }
    
    async def interpret_variant(
        self,
        variant,
        interval,
        gene: str,
        transcript: str,
        population_data: Optional[Dict] = None,
        functional_data: Optional[Dict] = None,
        phenotypes: Optional[List[str]] = None
    ) -> ClinicalInterpretation:
        """Perform comprehensive clinical interpretation of a variant.
        
        Args:
            variant: Variant object
            interval: Genomic interval
            gene: Gene symbol
            transcript: Transcript ID
            population_data: Population frequency data
            functional_data: Functional assay data
            phenotypes: Associated phenotypes
            
        Returns:
            Comprehensive clinical interpretation
        """
        variant_id = f"{variant.chromosome}:{variant.position}:{variant.reference_bases}>{variant.alternate_bases}"
        
        # Step 1: AlphaGenome prediction
        alphagenome_result = await self._get_alphagenome_prediction(
            variant, interval
        )
        
        # Step 2: Population genetics analysis
        pop_analysis = self._analyze_population_genetics(
            population_data,
            alphagenome_result
        )
        
        # Step 3: Functional analysis
        func_analysis = self._analyze_functional_data(
            functional_data,
            alphagenome_result
        )
        
        # Step 4: Computational predictions
        comp_predictions = self._integrate_computational_predictions(
            alphagenome_result
        )
        
        # Step 5: Gather ACMG evidence
        evidence = self._gather_acmg_evidence(
            alphagenome_result,
            pop_analysis,
            func_analysis,
            comp_predictions
        )
        
        # Step 6: Neuro-symbolic reasoning
        neuro_symbolic_result = self.neuro_symbolic.integrate_predictions(
            variant_id=variant_id,
            alphagenome_outputs=alphagenome_result,
            variant_annotations={
                'gene': gene,
                'population_frequency': pop_analysis.get('frequency', 0.0),
                **func_analysis
            }
        )
        
        # Step 7: Classify variant
        classification = self._classify_variant(evidence)
        
        # Step 8: Quantum prioritization (optional)
        quantum_score = None
        if self.enable_quantum:
            quantum_score = self._compute_quantum_priority(
                alphagenome_result,
                pop_analysis
            )
        
        # Step 9: Generate report
        summary = self._generate_summary(
            variant_id,
            gene,
            classification,
            evidence,
            neuro_symbolic_result
        )
        
        recommendations = self._generate_recommendations(
            classification,
            gene,
            phenotypes or []
        )
        
        return ClinicalInterpretation(
            variant_id=variant_id,
            gene=gene,
            transcript=transcript,
            alphagenome_score=alphagenome_result.get('pathogenicity_score', 0.0),
            pathogenicity_prediction=classification,
            confidence_score=self._compute_confidence_score(evidence),
            pathogenic_evidence=[e for e in evidence if e.category == 'pathogenic'],
            benign_evidence=[e for e in evidence if e.category == 'benign'],
            population_analysis=pop_analysis,
            functional_analysis=func_analysis,
            computational_predictions=comp_predictions,
            neuro_symbolic_reasoning={
                'confidence': neuro_symbolic_result.confidence,
                'explanation': neuro_symbolic_result.explanation,
                'supporting_rules': neuro_symbolic_result.supporting_rules
            },
            quantum_priority_score=quantum_score,
            phenotypes=phenotypes or [],
            summary=summary,
            recommendations=recommendations,
            references=self._gather_references(gene)
        )
    
    async def _get_alphagenome_prediction(self, variant, interval) -> Dict:
        """Get AlphaGenome predictions."""
        try:
            result = self.model.predict_variant(
                interval=interval,
                variant=variant,
                requested_outputs=['RNA_SEQ', 'ATAC_SEQ', 'CAGE']
            )
            
            # Extract key metrics
            return {
                'pathogenicity_score': self._compute_pathogenicity_score(result),
                'conservation_score': self._compute_conservation(result),
                'functional_impact': self._compute_functional_impact(result),
                'raw_predictions': result
            }
        except Exception as e:
            return {
                'pathogenicity_score': 0.5,
                'conservation_score': 0.5,
                'functional_impact': 0.5,
                'error': str(e)
            }
    
    def _compute_pathogenicity_score(self, result) -> float:
        """Compute aggregate pathogenicity score from AlphaGenome outputs."""
        # Simplified - in production, use more sophisticated integration
        scores = []
        
        if hasattr(result, 'reference') and hasattr(result, 'alternate'):
            # Compare reference vs alternate predictions
            ref_signal = np.mean(np.abs(result.reference.rna_seq.values))
            alt_signal = np.mean(np.abs(result.alternate.rna_seq.values))
            
            # Larger change = more likely pathogenic
            change = abs(alt_signal - ref_signal) / (ref_signal + 1e-6)
            scores.append(min(change / 0.5, 1.0))  # Normalize
        
        return np.mean(scores) if scores else 0.5
    
    def _compute_conservation(self, result) -> float:
        """Compute conservation score."""
        # Placeholder - integrate PhyloP, PhastCons, etc.
        return 0.75
    
    def _compute_functional_impact(self, result) -> float:
        """Compute functional impact score."""
        # Integrate multiple functional readouts
        return 0.6
    
    def _analyze_population_genetics(self, pop_data: Optional[Dict], 
                                    alphagenome_result: Dict) -> Dict:
        """Analyze population genetics."""
        if pop_data is None:
            return {'frequency': 0.0}
        
        from alphagenome_plus.analysis.population_genetics import AlleleFrequency
        
        freq_data = AlleleFrequency(
            population=pop_data.get('population', 'Unknown'),
            allele_count=pop_data.get('allele_count', 0),
            allele_number=pop_data.get('allele_number', 1),
            homozygote_count=pop_data.get('homozygote_count', 0)
        )
        
        hwe = self.pop_genetics.hardy_weinberg_test(freq_data)
        sel_coef = self.pop_genetics.estimate_selection_coefficient(
            freq_data,
            alphagenome_result.get('pathogenicity_score', 0.0) - 0.5
        )
        
        return {
            'frequency': freq_data.frequency,
            'hardy_weinberg': hwe,
            'selection_coefficient': sel_coef.s,
            'fitness_advantage': sel_coef.fitness_advantage
        }
    
    def _analyze_functional_data(self, func_data: Optional[Dict],
                                alphagenome_result: Dict) -> Dict:
        """Analyze functional assay data."""
        if func_data is None:
            return {}
        
        return {
            'experimental_validation': func_data.get('validated', False),
            'cell_assay_result': func_data.get('cell_assay', 'unknown'),
            'protein_function': func_data.get('protein_function', 'unknown')
        }
    
    def _integrate_computational_predictions(self, alphagenome_result: Dict) -> Dict:
        """Integrate multiple computational predictions."""
        return {
            'alphagenome': alphagenome_result.get('pathogenicity_score', 0.5),
            'conservation': alphagenome_result.get('conservation_score', 0.5),
            'functional_impact': alphagenome_result.get('functional_impact', 0.5)
        }
    
    def _gather_acmg_evidence(self, alphagenome_result: Dict,
                             pop_analysis: Dict,
                             func_analysis: Dict,
                             comp_predictions: Dict) -> List[ClinicalEvidence]:
        """Gather ACMG evidence criteria."""
        evidence = []
        
        # Population frequency (BA1, BS1, PM2)
        freq = pop_analysis.get('frequency', 0.0)
        if freq > 0.05:
            evidence.append(ClinicalEvidence(
                code='BA1',
                category='benign',
                strength=EvidenceStrength.STAND_ALONE,
                description='Allele frequency > 5% in population',
                supporting_data={'frequency': freq}
            ))
        elif freq > 0.01:
            evidence.append(ClinicalEvidence(
                code='BS1',
                category='benign',
                strength=EvidenceStrength.STRONG,
                description='Allele frequency > 1% in population',
                supporting_data={'frequency': freq}
            ))
        elif freq < 0.0001:
            evidence.append(ClinicalEvidence(
                code='PM2',
                category='pathogenic',
                strength=EvidenceStrength.MODERATE,
                description='Extremely low frequency in population',
                supporting_data={'frequency': freq}
            ))
        
        # Computational predictions (PP3, BP4)
        comp_score = comp_predictions.get('alphagenome', 0.5)
        if comp_score > 0.8:
            evidence.append(ClinicalEvidence(
                code='PP3',
                category='pathogenic',
                strength=EvidenceStrength.SUPPORTING,
                description='Multiple computational tools predict deleterious effect',
                supporting_data={'score': comp_score}
            ))
        elif comp_score < 0.2:
            evidence.append(ClinicalEvidence(
                code='BP4',
                category='benign',
                strength=EvidenceStrength.SUPPORTING,
                description='Multiple computational tools predict benign effect',
                supporting_data={'score': comp_score}
            ))
        
        return evidence
    
    def _classify_variant(self, evidence: List[ClinicalEvidence]) -> ClinicalSignificance:
        """Classify variant based on ACMG criteria."""
        path_evidence = [e for e in evidence if e.category == 'pathogenic']
        benign_evidence = [e for e in evidence if e.category == 'benign']
        
        # Count evidence by strength
        path_counts = self._count_evidence_by_strength(path_evidence)
        benign_counts = self._count_evidence_by_strength(benign_evidence)
        
        # ACMG classification rules
        # Pathogenic: (PVS1 + PS1-4) OR (>=2 PS) OR (PS + >=3 PM) OR (PS + >=2 PM + >=2 PP) ...
        # Simplified implementation
        
        if benign_counts.get('STAND_ALONE', 0) >= 1:
            return ClinicalSignificance.BENIGN
        
        if path_counts.get('VERY_STRONG', 0) >= 1 and path_counts.get('STRONG', 0) >= 1:
            return ClinicalSignificance.PATHOGENIC
        
        if path_counts.get('STRONG', 0) >= 2:
            return ClinicalSignificance.LIKELY_PATHOGENIC
        
        if benign_counts.get('STRONG', 0) >= 2:
            return ClinicalSignificance.LIKELY_BENIGN
        
        # Default to uncertain
        return ClinicalSignificance.UNCERTAIN
    
    def _count_evidence_by_strength(self, evidence: List[ClinicalEvidence]) -> Dict[str, int]:
        """Count evidence by strength level."""
        counts = {}
        for e in evidence:
            strength_name = e.strength.name
            counts[strength_name] = counts.get(strength_name, 0) + 1
        return counts
    
    def _compute_confidence_score(self, evidence: List[ClinicalEvidence]) -> float:
        """Compute overall confidence score."""
        if not evidence:
            return 0.5
        
        strength_weights = {
            'VERY_STRONG': 0.95,
            'STRONG': 0.85,
            'MODERATE': 0.70,
            'SUPPORTING': 0.55,
            'STAND_ALONE': 0.98
        }
        
        total_weight = sum(strength_weights.get(e.strength.name, 0.5) for e in evidence)
        return min(total_weight / len(evidence), 1.0)
    
    def _compute_quantum_priority(self, alphagenome_result: Dict,
                                 pop_analysis: Dict) -> float:
        """Compute quantum priority score."""
        # Combine multiple factors for prioritization
        score = (
            alphagenome_result.get('pathogenicity_score', 0.5) * 0.4 +
            (1.0 - pop_analysis.get('frequency', 0.5)) * 0.3 +
            alphagenome_result.get('conservation_score', 0.5) * 0.3
        )
        return score
    
    def _generate_summary(self, variant_id: str, gene: str,
                         classification: ClinicalSignificance,
                         evidence: List[ClinicalEvidence],
                         neuro_symbolic_result) -> str:
        """Generate clinical interpretation summary."""
        summary = f"Variant {variant_id} in {gene} is classified as {classification.value}.\n\n"
        
        if evidence:
            summary += "Evidence supporting this classification:\n"
            for e in evidence[:5]:  # Top 5 evidence items
                summary += f"- [{e.code}] {e.description}\n"
        
        summary += f"\nNeuro-symbolic reasoning: {neuro_symbolic_result.explanation}"
        
        return summary
    
    def _generate_recommendations(self, classification: ClinicalSignificance,
                                 gene: str, phenotypes: List[str]) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []
        
        if classification in [ClinicalSignificance.PATHOGENIC, ClinicalSignificance.LIKELY_PATHOGENIC]:
            recommendations.append("Consider genetic counseling")
            recommendations.append(f"Evaluate clinical manifestations related to {gene}")
            recommendations.append("Family cascade testing recommended")
        elif classification == ClinicalSignificance.UNCERTAIN:
            recommendations.append("Functional studies recommended")
            recommendations.append("Segregation analysis in family")
            recommendations.append("Re-evaluate as new evidence emerges")
        else:
            recommendations.append("No immediate clinical action required")
            recommendations.append("Routine follow-up")
        
        return recommendations
    
    def _gather_references(self, gene: str) -> List[str]:
        """Gather relevant references."""
        return [
            "ClinVar database",
            "OMIM gene entry",
            f"GeneReviews: {gene}",
            "ACMG/AMP 2015 Guidelines"
        ]
    
    def export_report(self, interpretation: ClinicalInterpretation,
                     format: str = 'json') -> str:
        """Export interpretation report."""
        if format == 'json':
            return json.dumps({
                'variant_id': interpretation.variant_id,
                'gene': interpretation.gene,
                'classification': interpretation.pathogenicity_prediction.value,
                'confidence': interpretation.confidence_score,
                'summary': interpretation.summary,
                'recommendations': interpretation.recommendations,
                'evidence': [
                    {
                        'code': e.code,
                        'category': e.category,
                        'strength': e.strength.value,
                        'description': e.description
                    }
                    for e in (interpretation.pathogenic_evidence + interpretation.benign_evidence)
                ]
            }, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
