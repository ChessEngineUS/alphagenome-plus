"""Integration with AlphaFold for structural impact prediction."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class StructuralImpact:
    """Structural impact assessment of a variant."""
    variant_id: str
    position: int
    reference_aa: str
    alternate_aa: str
    plddt_change: float
    buried: bool
    secondary_structure: str
    contacts_disrupted: int
    stability_score: float


class AlphaFoldStructureAnalyzer:
    """Analyze protein structures from AlphaFold to assess variant impact."""
    
    def __init__(self, cache_dir: str = "./alphafold_cache"):
        self.cache_dir = cache_dir
        self.base_url = "https://alphafold.ebi.ac.uk/api"
        
    def fetch_structure(self, uniprot_id: str) -> Dict:
        """Fetch AlphaFold structure for a protein.
        
        Args:
            uniprot_id: UniProt identifier
            
        Returns:
            Structure data dictionary
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required")
            
        url = f"{self.base_url}/prediction/{uniprot_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Failed to fetch structure for {uniprot_id}")
    
    def analyze_variant_impact(self,
                              uniprot_id: str,
                              position: int,
                              ref_aa: str,
                              alt_aa: str) -> StructuralImpact:
        """Analyze structural impact of a missense variant.
        
        Args:
            uniprot_id: UniProt identifier
            position: Amino acid position
            ref_aa: Reference amino acid
            alt_aa: Alternate amino acid
            
        Returns:
            Structural impact assessment
        """
        # Fetch structure
        structure = self.fetch_structure(uniprot_id)
        
        # Extract pLDDT scores
        plddt_scores = structure.get('plddt', [])
        if position <= len(plddt_scores):
            local_plddt = plddt_scores[position - 1]
        else:
            local_plddt = 50.0  # Unknown
            
        # Assess burial (simplified)
        buried = local_plddt > 70
        
        # Predict secondary structure change
        ss = self._predict_secondary_structure_change(ref_aa, alt_aa, position)
        
        # Estimate contacts disrupted
        contacts = self._estimate_contacts_disrupted(ref_aa, alt_aa)
        
        # Compute stability score
        stability = self._compute_stability_score(ref_aa, alt_aa, buried)
        
        return StructuralImpact(
            variant_id=f"{uniprot_id}:{ref_aa}{position}{alt_aa}",
            position=position,
            reference_aa=ref_aa,
            alternate_aa=alt_aa,
            plddt_change=-abs(self._aa_property_difference(ref_aa, alt_aa)) * 5,
            buried=buried,
            secondary_structure=ss,
            contacts_disrupted=contacts,
            stability_score=stability
        )
    
    def _predict_secondary_structure_change(self,
                                           ref_aa: str,
                                           alt_aa: str,
                                           position: int) -> str:
        """Predict secondary structure impact."""
        helix_formers = set('AELM')
        sheet_formers = set('VIFY')
        
        if ref_aa in helix_formers and alt_aa not in helix_formers:
            return 'helix_disruption'
        elif ref_aa in sheet_formers and alt_aa not in sheet_formers:
            return 'sheet_disruption'
        else:
            return 'preserved'
    
    def _estimate_contacts_disrupted(self, ref_aa: str, alt_aa: str) -> int:
        """Estimate number of contacts disrupted by substitution."""
        # Simple heuristic based on size difference
        aa_volumes = {
            'G': 60, 'A': 88, 'S': 89, 'C': 108, 'D': 111, 'P': 112,
            'N': 114, 'T': 116, 'E': 138, 'V': 140, 'Q': 143, 'H': 153,
            'M': 162, 'I': 166, 'L': 166, 'K': 168, 'R': 173, 'F': 189,
            'Y': 193, 'W': 227
        }
        
        volume_diff = abs(aa_volumes.get(ref_aa, 140) - aa_volumes.get(alt_aa, 140))
        return int(volume_diff / 30)  # Rough estimate
    
    def _compute_stability_score(self,
                                ref_aa: str,
                                alt_aa: str,
                                buried: bool) -> float:
        """Compute predicted stability change (ΔΔG)."""
        # Simplified Rosetta-like scoring
        hydrophobic = set('AILMFVWY')
        charged = set('DEKR')
        
        score = 0.0
        
        # Hydrophobic burial
        if buried:
            if ref_aa in hydrophobic and alt_aa not in hydrophobic:
                score -= 2.0
            elif ref_aa not in hydrophobic and alt_aa in hydrophobic:
                score += 1.0
        
        # Charge introduction
        if alt_aa in charged and ref_aa not in charged:
            score -= 1.5 if buried else -0.5
            
        # Proline effects
        if alt_aa == 'P':
            score -= 1.0
            
        return score
    
    def _aa_property_difference(self, aa1: str, aa2: str) -> float:
        """Compute property difference between amino acids."""
        properties = {
            'G': [0, 0, 0], 'A': [1, 0, 0], 'V': [3, 0, 0], 'L': [4, 0, 0],
            'I': [4, 0, 0], 'M': [4, 0, 0], 'F': [5, 0, 1], 'W': [6, 0, 1],
            'P': [2, 0, 0], 'S': [1, 1, 0], 'T': [2, 1, 0], 'C': [2, 1, 0],
            'Y': [5, 1, 1], 'N': [2, 1, 0], 'Q': [3, 1, 0], 'D': [3, -1, 0],
            'E': [4, -1, 0], 'K': [4, 1, 0], 'R': [5, 1, 1], 'H': [4, 0, 1]
        }
        
        prop1 = np.array(properties.get(aa1, [0, 0, 0]))
        prop2 = np.array(properties.get(aa2, [0, 0, 0]))
        
        return np.linalg.norm(prop1 - prop2)


class ESMEmbeddingAnalyzer:
    """Analyze ESM-2 embeddings for mutation effects."""
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Load ESM-2 model."""
        try:
            import torch
            import esm
            
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
        except ImportError:
            warnings.warn("ESM not available. Install with: pip install fair-esm")
            
    def compute_mutation_score(self,
                             sequence: str,
                             position: int,
                             alt_aa: str) -> float:
        """Compute mutation effect score using ESM-2.
        
        Args:
            sequence: Wild-type protein sequence
            position: Mutation position (0-indexed)
            alt_aa: Alternate amino acid
            
        Returns:
            Mutation effect score (higher = more deleterious)
        """
        if self.model is None:
            self.load_model()
            
        import torch
        
        # Prepare sequences
        data = [
            ("wildtype", sequence),
            ("mutant", sequence[:position] + alt_aa + sequence[position+1:])
        ]
        
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33]
            
        # Compute embedding difference
        wt_emb = embeddings[0, position+1, :]
        mt_emb = embeddings[1, position+1, :]
        
        score = torch.nn.functional.cosine_similarity(wt_emb, mt_emb, dim=0)
        
        return 1.0 - score.item()  # Higher score = more different = more deleterious
