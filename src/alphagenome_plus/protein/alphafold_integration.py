"""Integration with AlphaFold for protein structure-based variant analysis.

Combines AlphaGenome sequence predictions with AlphaFold structural 
predictions to assess variant impacts on protein structure and function.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from Bio.PDB import PDBParser, DSSP, PPBuilder
from Bio import SeqIO
from pathlib import Path


@dataclass
class StructuralImpact:
    """Container for variant structural impact assessment."""
    variant_id: str
    position: int
    ref_aa: str
    alt_aa: str
    
    # Structural metrics
    plddt_change: float  # Change in pLDDT confidence score
    secondary_structure_change: bool
    buried_surface_change: float  # Change in buried surface area
    binding_site_distance: float  # Distance to nearest binding site
    
    # Functional predictions
    destabilizing_score: float  # 0-1, higher = more destabilizing
    functional_impact_score: float  # Combined impact score


class AlphaFoldIntegration:
    """Integrate AlphaGenome with AlphaFold for structural analysis.
    
    Args:
        alphafold_api_url: URL for AlphaFold database API
        cache_dir: Directory to cache structure files
    """
    
    def __init__(self, alphafold_api_url: str = "https://alphafold.ebi.ac.uk/api",
                 cache_dir: Optional[Path] = None):
        self.api_url = alphafold_api_url
        self.cache_dir = cache_dir or Path.home() / ".alphagenome_plus" / "structures"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_parser = PDBParser(QUIET=True)
    
    def fetch_structure(self, uniprot_id: str) -> Optional[Path]:
        """Fetch AlphaFold structure from database.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            Path to downloaded PDB file, or None if not found
        """
        cache_path = self.cache_dir / f"{uniprot_id}.pdb"
        
        # Check cache first
        if cache_path.exists():
            return cache_path
        
        # Download from AlphaFold DB
        url = f"{self.api_url}/prediction/{uniprot_id}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse JSON response to get PDB URL
            data = response.json()
            pdb_url = data[0]['pdbUrl']
            
            # Download PDB file
            pdb_response = requests.get(pdb_url)
            pdb_response.raise_for_status()
            
            # Save to cache
            with open(cache_path, 'w') as f:
                f.write(pdb_response.text)
            
            return cache_path
            
        except (requests.RequestException, KeyError, IndexError) as e:
            print(f"Failed to fetch structure for {uniprot_id}: {e}")
            return None
    
    def calculate_plddt(self, structure_path: Path) -> Dict[int, float]:
        """Extract per-residue pLDDT confidence scores.
        
        Args:
            structure_path: Path to PDB file
            
        Returns:
            Dictionary mapping residue number to pLDDT score
        """
        structure = self.pdb_parser.get_structure('protein', structure_path)
        plddt_scores = {}
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # pLDDT is stored in B-factor field for AlphaFold
                    res_num = residue.get_id()[1]
                    atoms = list(residue.get_atoms())
                    if atoms:
                        plddt_scores[res_num] = atoms[0].get_bfactor()
        
        return plddt_scores
    
    def get_secondary_structure(self, structure_path: Path) -> Dict[int, str]:
        """Determine secondary structure for each residue.
        
        Args:
            structure_path: Path to PDB file
            
        Returns:
            Dictionary mapping residue number to secondary structure 
            (H=helix, E=sheet, C=coil)
        """
        structure = self.pdb_parser.get_structure('protein', structure_path)
        model = structure[0]
        
        # Use DSSP for secondary structure assignment
        try:
            dssp = DSSP(model, str(structure_path), dssp='mkdssp')
            
            ss_dict = {}
            for key in dssp:
                residue_num = key[1][1]
                ss_code = dssp[key][2]
                
                # Simplify to 3-state: H, E, C
                if ss_code in ['H', 'G', 'I']:
                    ss = 'H'  # Helix
                elif ss_code in ['E', 'B']:
                    ss = 'E'  # Sheet  
                else:
                    ss = 'C'  # Coil
                
                ss_dict[residue_num] = ss
            
            return ss_dict
            
        except Exception as e:
            print(f"DSSP failed: {e}. Using simple assignment.")
            # Fallback: rough assignment based on phi/psi angles
            return self._simple_ss_assignment(structure_path)
    
    def _simple_ss_assignment(self, structure_path: Path) -> Dict[int, str]:
        """Simple secondary structure assignment without DSSP."""
        structure = self.pdb_parser.get_structure('protein', structure_path)
        ppb = PPBuilder()
        
        ss_dict = {}
        for pp in ppb.build_peptides(structure):
            phi_psi = pp.get_phi_psi_list()
            for i, (phi, psi) in enumerate(phi_psi):
                if phi is None or psi is None:
                    ss_dict[i+1] = 'C'
                    continue
                
                # Simple classification
                if -90 <= phi <= -30 and -70 <= psi <= -10:
                    ss_dict[i+1] = 'H'  # Alpha helix
                elif -150 <= phi <= -90 and 90 <= psi <= 150:
                    ss_dict[i+1] = 'E'  # Beta sheet
                else:
                    ss_dict[i+1] = 'C'  # Coil
        
        return ss_dict
    
    def calculate_buried_surface(self, structure_path: Path, 
                                position: int) -> float:
        """Calculate buried surface area at position.
        
        Args:
            structure_path: Path to PDB file
            position: Residue position
            
        Returns:
            Approximate buried surface area in Ų
        """
        structure = self.pdb_parser.get_structure('protein', structure_path)
        
        # Find target residue
        target_residue = None
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[1] == position:
                        target_residue = residue
                        break
        
        if target_residue is None:
            return 0.0
        
        # Count nearby atoms (simple burial approximation)
        target_coords = [atom.get_coord() for atom in target_residue]
        if not target_coords:
            return 0.0
        
        center = np.mean(target_coords, axis=0)
        
        # Count atoms within 8Å sphere
        nearby_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[1] == position:
                        continue
                    for atom in residue:
                        dist = np.linalg.norm(atom.get_coord() - center)
                        if dist < 8.0:
                            nearby_count += 1
        
        # Rough estimate: more nearby atoms = more buried
        return nearby_count * 15.0  # Approximate Ų per contact
    
    def assess_variant_impact(self, uniprot_id: str, position: int,
                             ref_aa: str, alt_aa: str,
                             variant_id: str) -> StructuralImpact:
        """Assess structural impact of amino acid variant.
        
        Args:
            uniprot_id: UniProt ID for protein
            position: Amino acid position
            ref_aa: Reference amino acid (single letter)
            alt_aa: Alternate amino acid (single letter)
            variant_id: Variant identifier
            
        Returns:
            StructuralImpact assessment
        """
        # Fetch structure
        structure_path = self.fetch_structure(uniprot_id)
        if structure_path is None:
            raise ValueError(f"Could not fetch structure for {uniprot_id}")
        
        # Get structural metrics
        plddt_scores = self.calculate_plddt(structure_path)
        ss_assignment = self.get_secondary_structure(structure_path)
        buried_surface = self.calculate_buried_surface(structure_path, position)
        
        # Calculate changes (simplified - would need actual mutation modeling)
        plddt_change = plddt_scores.get(position, 50.0) - 70.0  # Baseline
        
        # Estimate destabilization based on substitution matrix
        destabilizing = self._estimate_destabilization(ref_aa, alt_aa, 
                                                       ss_assignment.get(position, 'C'))
        
        # Combine metrics for functional impact
        functional_impact = (
            0.4 * destabilizing +
            0.3 * (1.0 - plddt_scores.get(position, 50.0) / 100.0) +
            0.3 * min(buried_surface / 500.0, 1.0)
        )
        
        return StructuralImpact(
            variant_id=variant_id,
            position=position,
            ref_aa=ref_aa,
            alt_aa=alt_aa,
            plddt_change=plddt_change,
            secondary_structure_change=False,  # Would need mutation modeling
            buried_surface_change=0.0,  # Would need mutation modeling
            binding_site_distance=0.0,  # Would need ligand/interface data
            destabilizing_score=destabilizing,
            functional_impact_score=functional_impact
        )
    
    def _estimate_destabilization(self, ref_aa: str, alt_aa: str, 
                                 ss_context: str) -> float:
        """Estimate destabilization score for amino acid substitution.
        
        Returns:
            Score from 0 (stabilizing) to 1 (highly destabilizing)
        """
        # Simplified BLOSUM-like scoring
        hydrophobic = set(['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'])
        charged = set(['K', 'R', 'D', 'E'])
        polar = set(['S', 'T', 'N', 'Q', 'Y', 'C'])
        
        # Conservative substitutions are less destabilizing
        if ref_aa == alt_aa:
            return 0.0
        
        ref_class = (
            'hydrophobic' if ref_aa in hydrophobic else
            'charged' if ref_aa in charged else 'polar'
        )
        alt_class = (
            'hydrophobic' if alt_aa in hydrophobic else
            'charged' if alt_aa in charged else 'polar'
        )
        
        # Class switch penalties
        if ref_class == alt_class:
            base_score = 0.2
        elif {ref_class, alt_class} == {'hydrophobic', 'polar'}:
            base_score = 0.5
        else:
            base_score = 0.8  # Hydrophobic<->charged very destabilizing
        
        # Context-dependent penalties
        if ss_context == 'H' and alt_aa == 'P':
            base_score += 0.3  # Proline breaks helices
        elif ss_context == 'E' and alt_aa in charged:
            base_score += 0.2  # Charged residues disrupt sheets
        
        return min(base_score, 1.0)
