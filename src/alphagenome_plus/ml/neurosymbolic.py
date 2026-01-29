"""Neuro-symbolic AI integration for genomic variant interpretation.

Combines neural network predictions from AlphaGenome with symbolic reasoning
using logical rules, ontologies, and constraint satisfaction.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict


class LogicalOperator(Enum):
    """Logical operators for rule-based reasoning."""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if


@dataclass
class BiologicalConcept:
    """Represents a biological concept or entity."""
    name: str
    ontology_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    
    def has_property(self, prop: str, value: Any = None) -> bool:
        """Check if concept has a property with optional value check."""
        if prop not in self.properties:
            return False
        if value is None:
            return True
        return self.properties[prop] == value
    
    def related_to(self, relation: str, other: str) -> bool:
        """Check if concept has relationship to another concept."""
        return relation in self.relationships and other in self.relationships[relation]


@dataclass
class LogicalRule:
    """Symbolic rule for genomic reasoning."""
    name: str
    conditions: List[Tuple[str, Any]]  # (condition_expr, expected_value)
    conclusion: Tuple[str, Any]  # (conclusion_expr, value)
    confidence: float = 1.0
    priority: int = 0
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate rule against context.
        
        Returns:
            (rule_fires, confidence_score)
        """
        # Check all conditions
        for condition, expected in self.conditions:
            value = self._evaluate_expression(condition, context)
            if value != expected:
                return False, 0.0
        
        return True, self.confidence
    
    def _evaluate_expression(self, expr: str, context: Dict[str, Any]) -> Any:
        """Evaluate a logical expression."""
        # Simple evaluation - in production, use a proper parser
        # Handle basic comparisons and property access
        for key, value in context.items():
            expr = expr.replace(key, str(value))
        
        try:
            return eval(expr, {"__builtins__": {}}, context)
        except:
            return False


@dataclass
class VariantInterpretation:
    """Neuro-symbolic interpretation of variant."""
    variant_id: str
    neural_predictions: Dict[str, float]
    symbolic_inferences: List[str]
    confidence: float
    explanation: str
    supporting_rules: List[str]
    contradictions: List[str] = field(default_factory=list)


class KnowledgeBase:
    """Knowledge base for biological concepts and relationships."""
    
    def __init__(self):
        self.concepts: Dict[str, BiologicalConcept] = {}
        self.rules: List[LogicalRule] = []
        self._init_default_knowledge()
    
    def _init_default_knowledge(self):
        """Initialize with basic genomic knowledge."""
        # Add basic concepts
        self.add_concept(BiologicalConcept(
            name="missense_variant",
            ontology_id="SO:0001583",
            properties={
                "alters_protein": True,
                "severity": "moderate"
            }
        ))
        
        self.add_concept(BiologicalConcept(
            name="splice_site",
            ontology_id="SO:0000162",
            properties={
                "functional_importance": "high",
                "conservation_required": True
            }
        ))
        
        self.add_concept(BiologicalConcept(
            name="loss_of_function",
            properties={
                "pathogenic_potential": "high",
                "requires_haploinsufficiency_check": True
            },
            relationships={
                "causes": ["protein_truncation", "splice_disruption"]
            }
        ))
        
        # Add inference rules
        self.add_rule(LogicalRule(
            name="high_conservation_pathogenic",
            conditions=[
                ("conservation_score > 0.9", True),
                ("predicted_damaging > 0.8", True)
            ],
            conclusion=("pathogenic_likelihood", "high"),
            confidence=0.85,
            priority=10
        ))
        
        self.add_rule(LogicalRule(
            name="common_benign",
            conditions=[
                ("population_frequency > 0.01", True),
                ("predicted_damaging < 0.5", True)
            ],
            conclusion=("pathogenic_likelihood", "low"),
            confidence=0.9,
            priority=15
        ))
        
        self.add_rule(LogicalRule(
            name="splice_disruption_severe",
            conditions=[
                ("variant_type == 'splice_site'", True),
                ("splice_ai_score > 0.5", True)
            ],
            conclusion=("functional_impact", "high"),
            confidence=0.92,
            priority=12
        ))
    
    def add_concept(self, concept: BiologicalConcept):
        """Add biological concept to knowledge base."""
        self.concepts[concept.name] = concept
    
    def add_rule(self, rule: LogicalRule):
        """Add logical rule to knowledge base."""
        self.rules.append(rule)
        # Sort by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def get_concept(self, name: str) -> Optional[BiologicalConcept]:
        """Retrieve concept by name."""
        return self.concepts.get(name)
    
    def query_relationships(self, subject: str, relation: str) -> List[str]:
        """Query relationships in knowledge graph."""
        concept = self.get_concept(subject)
        if concept and relation in concept.relationships:
            return concept.relationships[relation]
        return []


class NeuroSymbolicReasoner:
    """Combines neural predictions with symbolic reasoning."""
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.kb = knowledge_base or KnowledgeBase()
        self.explanation_templates = self._init_explanation_templates()
    
    def _init_explanation_templates(self) -> Dict[str, str]:
        """Initialize explanation templates for interpretations."""
        return {
            "high_pathogenic": "Variant is likely pathogenic due to {reasons}.",
            "low_pathogenic": "Variant is likely benign because {reasons}.",
            "uncertain": "Variant has uncertain significance. {reasons}",
            "conflicting": "Evidence is conflicting: {reasons}"
        }
    
    def integrate_predictions(self,
                            variant_id: str,
                            alphagenome_outputs: Dict[str, float],
                            variant_annotations: Dict[str, Any]) -> VariantInterpretation:
        """Integrate neural predictions with symbolic reasoning.
        
        Args:
            variant_id: Variant identifier
            alphagenome_outputs: Neural network predictions from AlphaGenome
            variant_annotations: Variant annotations (type, location, etc.)
            
        Returns:
            Integrated neuro-symbolic interpretation
        """
        # Build context for rule evaluation
        context = {
            **alphagenome_outputs,
            **variant_annotations
        }
        
        # Add derived features
        context['predicted_damaging'] = alphagenome_outputs.get('pathogenicity_score', 0.0)
        context['conservation_score'] = alphagenome_outputs.get('conservation', 0.0)
        
        # Apply symbolic rules
        fired_rules = []
        symbolic_conclusions = []
        total_confidence = 0.0
        
        for rule in self.kb.rules:
            fires, conf = rule.evaluate(context)
            if fires:
                fired_rules.append(rule.name)
                symbolic_conclusions.append(rule.conclusion)
                total_confidence += conf
        
        # Reconcile neural and symbolic evidence
        neural_pathogenic = alphagenome_outputs.get('pathogenicity_score', 0.0)
        
        # Count symbolic votes
        pathogenic_votes = sum(1 for _, val in symbolic_conclusions if val in ['high', 'pathogenic'])
        benign_votes = sum(1 for _, val in symbolic_conclusions if val in ['low', 'benign'])
        
        # Compute integrated confidence
        if len(fired_rules) > 0:
            symbolic_confidence = total_confidence / len(fired_rules)
            # Weighted combination: 60% neural, 40% symbolic
            integrated_confidence = 0.6 * neural_pathogenic + 0.4 * symbolic_confidence
        else:
            integrated_confidence = neural_pathogenic
        
        # Detect contradictions
        contradictions = []
        if neural_pathogenic > 0.7 and benign_votes > pathogenic_votes:
            contradictions.append("Neural prediction suggests pathogenic but symbolic rules suggest benign")
        elif neural_pathogenic < 0.3 and pathogenic_votes > benign_votes:
            contradictions.append("Neural prediction suggests benign but symbolic rules suggest pathogenic")
        
        # Generate explanation
        explanation = self._generate_explanation(
            neural_pathogenic,
            symbolic_conclusions,
            fired_rules,
            contradictions
        )
        
        # Compile inferences
        inferences = [f"{name}: {value}" for name, value in symbolic_conclusions]
        
        return VariantInterpretation(
            variant_id=variant_id,
            neural_predictions=alphagenome_outputs,
            symbolic_inferences=inferences,
            confidence=integrated_confidence,
            explanation=explanation,
            supporting_rules=fired_rules,
            contradictions=contradictions
        )
    
    def _generate_explanation(self,
                            neural_score: float,
                            symbolic_conclusions: List[Tuple[str, Any]],
                            fired_rules: List[str],
                            contradictions: List[str]) -> str:
        """Generate human-readable explanation."""
        reasons = []
        
        # Neural evidence
        if neural_score > 0.7:
            reasons.append(f"AlphaGenome predicts high pathogenicity (score: {neural_score:.2f})")
        elif neural_score < 0.3:
            reasons.append(f"AlphaGenome predicts low pathogenicity (score: {neural_score:.2f})")
        else:
            reasons.append(f"AlphaGenome prediction is uncertain (score: {neural_score:.2f})")
        
        # Symbolic evidence
        if len(fired_rules) > 0:
            reasons.append(f"Supported by {len(fired_rules)} logical rule(s): {', '.join(fired_rules[:3])}")
        
        # Add key conclusions
        for name, value in symbolic_conclusions[:2]:
            reasons.append(f"{name.replace('_', ' ')}: {value}")
        
        # Note contradictions
        if contradictions:
            return self.explanation_templates['conflicting'].format(
                reasons=' AND '.join(reasons) + f". However, {contradictions[0]}"
            )
        
        # Determine template
        if neural_score > 0.6 or any(v in ['high', 'pathogenic'] for _, v in symbolic_conclusions):
            template = self.explanation_templates['high_pathogenic']
        elif neural_score < 0.4 or any(v in ['low', 'benign'] for _, v in symbolic_conclusions):
            template = self.explanation_templates['low_pathogenic']
        else:
            template = self.explanation_templates['uncertain']
        
        return template.format(reasons='; '.join(reasons))
    
    def add_domain_knowledge(self, rules: List[LogicalRule]):
        """Add custom domain-specific rules."""
        for rule in rules:
            self.kb.add_rule(rule)
    
    def explain_reasoning(self, interpretation: VariantInterpretation) -> str:
        """Generate detailed reasoning trace."""
        trace = []
        trace.append(f"\n=== Reasoning Trace for {interpretation.variant_id} ===")
        trace.append(f"\nNeural Predictions:")
        for key, value in interpretation.neural_predictions.items():
            trace.append(f"  - {key}: {value:.3f}")
        
        trace.append(f"\nSymbolic Inferences ({len(interpretation.symbolic_inferences)}):")
        for inference in interpretation.symbolic_inferences:
            trace.append(f"  - {inference}")
        
        trace.append(f"\nSupporting Rules ({len(interpretation.supporting_rules)}):")
        for rule_name in interpretation.supporting_rules:
            trace.append(f"  - {rule_name}")
        
        if interpretation.contradictions:
            trace.append(f"\nContradictions Detected:")
            for contradiction in interpretation.contradictions:
                trace.append(f"  ! {contradiction}")
        
        trace.append(f"\nFinal Confidence: {interpretation.confidence:.2f}")
        trace.append(f"\nExplanation: {interpretation.explanation}")
        
        return '\n'.join(trace)


class OntologyIntegrator:
    """Integrates genomic ontologies for enhanced reasoning."""
    
    def __init__(self):
        self.ontology_terms: Dict[str, BiologicalConcept] = {}
        self.parent_child_relations: Dict[str, List[str]] = defaultdict(list)
        self._load_sequence_ontology_subset()
    
    def _load_sequence_ontology_subset(self):
        """Load subset of Sequence Ontology terms."""
        # Key SO terms for variant annotation
        so_terms = [
            ("SO:0001483", "SNV", {"molecular_type": "substitution"}),
            ("SO:0001587", "stop_gained", {"severity": "high", "lof": True}),
            ("SO:0001589", "frameshift_variant", {"severity": "high", "lof": True}),
            ("SO:0001575", "splice_donor_variant", {"severity": "high"}),
            ("SO:0001574", "splice_acceptor_variant", {"severity": "high"}),
            ("SO:0001583", "missense_variant", {"severity": "moderate"}),
            ("SO:0001819", "synonymous_variant", {"severity": "low"}),
        ]
        
        for onto_id, name, props in so_terms:
            concept = BiologicalConcept(
                name=name,
                ontology_id=onto_id,
                properties=props
            )
            self.ontology_terms[onto_id] = concept
            self.ontology_terms[name] = concept
    
    def get_severity(self, variant_type: str) -> str:
        """Get severity classification from ontology."""
        concept = self.ontology_terms.get(variant_type)
        if concept and 'severity' in concept.properties:
            return concept.properties['severity']
        return "unknown"
    
    def is_loss_of_function(self, variant_type: str) -> bool:
        """Check if variant type is loss-of-function."""
        concept = self.ontology_terms.get(variant_type)
        return concept is not None and concept.properties.get('lof', False)


def create_clinical_reasoning_system() -> NeuroSymbolicReasoner:
    """Factory function to create a clinical-grade reasoning system."""
    kb = KnowledgeBase()
    
    # Add clinical interpretation rules (ACMG-like)
    clinical_rules = [
        LogicalRule(
            name="ACMG_PVS1",  # Very strong pathogenic
            conditions=[
                ("variant_type in ['stop_gained', 'frameshift_variant']", True),
                ("gene_haploinsufficiency == True", True)
            ],
            conclusion=("pathogenic_evidence", "very_strong"),
            confidence=0.95,
            priority=20
        ),
        LogicalRule(
            name="ACMG_PM2",  # Moderate pathogenic
            conditions=[
                ("population_frequency < 0.0001", True),
                ("predicted_damaging > 0.7", True)
            ],
            conclusion=("pathogenic_evidence", "moderate"),
            confidence=0.75,
            priority=8
        ),
        LogicalRule(
            name="ACMG_BA1",  # Stand-alone benign
            conditions=[
                ("population_frequency > 0.05", True)
            ],
            conclusion=("benign_evidence", "stand_alone"),
            confidence=0.98,
            priority=25
        ),
    ]
    
    for rule in clinical_rules:
        kb.add_rule(rule)
    
    return NeuroSymbolicReasoner(knowledge_base=kb)
