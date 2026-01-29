"""Population genetics analysis for variant interpretation.

This module provides tools for integrating allele frequency data with
AlphaGenome predictions to estimate selection coefficients, identify
balancing selection signatures, and perform Hardy-Weinberg equilibrium tests.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import scipy.stats as stats
from scipy.optimize import minimize


@dataclass
class AlleleFrequency:
    """Allele frequency data from population databases."""
    population: str
    allele_count: int
    allele_number: int
    homozygote_count: int
    
    @property
    def frequency(self) -> float:
        """Calculate allele frequency."""
        return self.allele_count / self.allele_number if self.allele_number > 0 else 0.0
    
    @property
    def heterozygote_count(self) -> int:
        """Estimate heterozygote count."""
        return self.allele_count - 2 * self.homozygote_count


@dataclass
class SelectionCoefficient:
    """Selection coefficient estimation result."""
    s: float  # Selection coefficient
    h: float  # Dominance coefficient
    confidence_interval: Tuple[float, float]
    p_value: float
    fitness_advantage: float


class PopulationGeneticsAnalyzer:
    """Analyze variant effects in population genetics context."""
    
    def __init__(self, 
                 effective_population_size: int = 10000,
                 mutation_rate: float = 1e-8,
                 generation_time: float = 25.0):
        """Initialize analyzer.
        
        Args:
            effective_population_size: Effective population size
            mutation_rate: Per-base mutation rate per generation
            generation_time: Generation time in years
        """
        self.Ne = effective_population_size
        self.mu = mutation_rate
        self.generation_time = generation_time
    
    def hardy_weinberg_test(self, 
                           freq_data: AlleleFrequency,
                           alpha: float = 0.05) -> Dict[str, Union[float, bool]]:
        """Perform Hardy-Weinberg equilibrium test.
        
        Args:
            freq_data: Allele frequency data
            alpha: Significance level
            
        Returns:
            Dictionary with test statistics and results
        """
        p = freq_data.frequency
        q = 1 - p
        n = freq_data.allele_number // 2  # Number of individuals
        
        # Expected counts under HWE
        expected_AA = n * p * p
        expected_Aa = n * 2 * p * q
        expected_aa = n * q * q
        
        # Observed counts
        observed_AA = freq_data.homozygote_count
        observed_Aa = freq_data.heterozygote_count
        observed_aa = n - observed_AA - observed_Aa
        
        # Chi-square test
        chi2 = 0.0
        for obs, exp in [(observed_AA, expected_AA), 
                         (observed_Aa, expected_Aa), 
                         (observed_aa, expected_aa)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        
        # Degrees of freedom = 1 for HWE test
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        # Calculate inbreeding coefficient F
        F = 1 - (observed_Aa / expected_Aa) if expected_Aa > 0 else 0.0
        
        return {
            'chi_square': chi2,
            'p_value': p_value,
            'in_equilibrium': p_value > alpha,
            'inbreeding_coefficient': F,
            'heterozygote_excess': observed_Aa > expected_Aa
        }
    
    def estimate_selection_coefficient(self,
                                      freq_data: AlleleFrequency,
                                      predicted_fitness_effect: float,
                                      time_generations: Optional[int] = None) -> SelectionCoefficient:
        """Estimate selection coefficient from allele frequency and fitness effect.
        
        Uses diffusion approximation and maximum likelihood estimation.
        
        Args:
            freq_data: Current allele frequency data
            predicted_fitness_effect: Fitness effect from AlphaGenome (-1 to 1)
            time_generations: Time in generations (defaults to Ne/10)
            
        Returns:
            Estimated selection coefficient and statistics
        """
        p = freq_data.frequency
        
        if time_generations is None:
            time_generations = self.Ne // 10
        
        # Initial guess based on predicted effect
        s_init = predicted_fitness_effect * 0.01  # Scale to reasonable selection coefficient
        
        def negative_log_likelihood(params):
            """Negative log-likelihood for selection coefficient."""
            s, h = params
            
            # Prevent extreme values
            if abs(s) > 1.0 or h < 0 or h > 1.0:
                return 1e10
            
            # Diffusion approximation for allele frequency
            # Simplified Wright-Fisher model
            Ne_s = self.Ne * s
            
            # Expected frequency change
            delta_p = s * p * (1 - p) * (h + (1 - 2*h) * p)
            
            # Variance in frequency change (drift)
            var_p = p * (1 - p) / (2 * self.Ne)
            
            # Likelihood of observing current frequency
            if var_p <= 0:
                return 1e10
            
            # Gaussian approximation
            expected = delta_p * time_generations
            nll = 0.5 * (expected ** 2) / (var_p * time_generations)
            
            return nll
        
        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0=[s_init, 0.5],  # Start with additive (h=0.5)
            bounds=[(-0.5, 0.5), (0.0, 1.0)],
            method='L-BFGS-B'
        )
        
        s_mle, h_mle = result.x
        
        # Approximate confidence interval using Fisher information
        # Simplified: use Hessian inverse
        try:
            hessian_inv = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else np.array([[1.0]])
            se = np.sqrt(np.diag(hessian_inv)[0]) if hessian_inv.shape[0] > 0 else 0.01
        except:
            se = 0.01  # Default standard error
        
        ci = (s_mle - 1.96 * se, s_mle + 1.96 * se)
        
        # Calculate p-value for neutrality test
        z_score = s_mle / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Fitness advantage
        fitness_advantage = 2 * self.Ne * s_mle
        
        return SelectionCoefficient(
            s=s_mle,
            h=h_mle,
            confidence_interval=ci,
            p_value=p_value,
            fitness_advantage=fitness_advantage
        )
    
    def detect_balancing_selection(self,
                                  freq_data: List[AlleleFrequency],
                                  min_populations: int = 3) -> Dict[str, Union[bool, float, List[str]]]:
        """Detect signatures of balancing selection.
        
        Balancing selection maintains intermediate allele frequencies across populations.
        
        Args:
            freq_data: Allele frequency data from multiple populations
            min_populations: Minimum populations with intermediate frequency
            
        Returns:
            Dictionary with balancing selection analysis results
        """
        # Intermediate frequency range (typical for balancing selection)
        intermediate_min = 0.2
        intermediate_max = 0.8
        
        intermediate_pops = []
        frequencies = []
        
        for data in freq_data:
            f = data.frequency
            frequencies.append(f)
            
            if intermediate_min <= f <= intermediate_max:
                intermediate_pops.append(data.population)
        
        # Calculate variance in frequencies across populations
        freq_variance = np.var(frequencies) if len(frequencies) > 1 else 0.0
        
        # Calculate Fst (genetic differentiation)
        mean_freq = np.mean(frequencies)
        p_bar = mean_freq
        q_bar = 1 - p_bar
        
        # Expected heterozygosity
        H_t = 2 * p_bar * q_bar
        
        # Average within-population heterozygosity
        H_s = np.mean([2 * f * (1 - f) for f in frequencies])
        
        # Fst
        F_st = (H_t - H_s) / H_t if H_t > 0 else 0.0
        
        # Balancing selection signature: 
        # - Multiple populations with intermediate frequency
        # - Low Fst (similar across populations)
        # - Low variance
        is_balancing = (
            len(intermediate_pops) >= min_populations and
            F_st < 0.15 and  # Low differentiation
            freq_variance < 0.05  # Low variance
        )
        
        return {
            'is_balancing_selection': is_balancing,
            'intermediate_frequency_populations': intermediate_pops,
            'num_intermediate': len(intermediate_pops),
            'f_st': F_st,
            'frequency_variance': freq_variance,
            'mean_frequency': mean_freq,
            'confidence': len(intermediate_pops) / len(freq_data) if len(freq_data) > 0 else 0.0
        }
    
    def estimate_allele_age(self,
                          freq_data: AlleleFrequency,
                          selection_coef: float = 0.0) -> Dict[str, float]:
        """Estimate age of allele using frequency and selection coefficient.
        
        Uses analytical approximation for allele age.
        
        Args:
            freq_data: Current allele frequency
            selection_coef: Selection coefficient (0 for neutral)
            
        Returns:
            Dictionary with age estimates in generations and years
        """
        p = freq_data.frequency
        
        if p <= 0 or p >= 1:
            return {'age_generations': float('inf'), 'age_years': float('inf')}
        
        if abs(selection_coef) < 1e-6:
            # Neutral allele age approximation
            # E[T] ≈ -4*Ne * (p*log(p) + (1-p)*log(1-p))
            if p > 0 and p < 1:
                age_gen = -4 * self.Ne * (p * np.log(p) + (1 - p) * np.log(1 - p))
            else:
                age_gen = 0.0
        else:
            # Selected allele
            # Approximate using deterministic trajectory
            Ne_s = self.Ne * selection_coef
            
            if Ne_s > 0:
                # Advantageous allele
                age_gen = np.log(p / (1 - p)) / selection_coef
            else:
                # Deleterious allele (likely recent)
                age_gen = -np.log(p) / abs(selection_coef)
        
        age_years = age_gen * self.generation_time
        
        return {
            'age_generations': max(0, age_gen),
            'age_years': max(0, age_years),
            'age_confidence': 'low' if abs(selection_coef) > 0.01 else 'moderate'
        }


def integrate_with_alphagenome_predictions(alphagenome_effect: float,
                                          freq_data: AlleleFrequency,
                                          analyzer: Optional[PopulationGeneticsAnalyzer] = None) -> Dict:
    """Integrate AlphaGenome predictions with population genetics analysis.
    
    Args:
        alphagenome_effect: Predicted fitness effect from AlphaGenome
        freq_data: Population allele frequency data
        analyzer: PopulationGeneticsAnalyzer instance
        
    Returns:
        Comprehensive analysis combining genomic and population data
    """
    if analyzer is None:
        analyzer = PopulationGeneticsAnalyzer()
    
    # HWE test
    hwe_result = analyzer.hardy_weinberg_test(freq_data)
    
    # Selection coefficient estimation
    sel_coef = analyzer.estimate_selection_coefficient(freq_data, alphagenome_effect)
    
    # Allele age
    age = analyzer.estimate_allele_age(freq_data, sel_coef.s)
    
    # Interpret results
    interpretation = "neutral"
    if sel_coef.p_value < 0.05:
        if sel_coef.s > 0:
            interpretation = "advantageous"
        else:
            interpretation = "deleterious"
    
    # Clinical significance prediction
    clinical_significance = "uncertain"
    if abs(alphagenome_effect) > 0.5 and abs(sel_coef.s) > 0.001:
        if sel_coef.s < -0.005:
            clinical_significance = "likely pathogenic"
        elif freq_data.frequency < 0.001:
            clinical_significance = "possibly pathogenic"
    elif freq_data.frequency > 0.05 and not hwe_result['in_equilibrium']:
        clinical_significance = "likely benign or balanced"
    
    return {
        'alphagenome_effect': alphagenome_effect,
        'population_frequency': freq_data.frequency,
        'hardy_weinberg': hwe_result,
        'selection_coefficient': sel_coef,
        'allele_age': age,
        'interpretation': interpretation,
        'clinical_significance': clinical_significance,
        'confidence_score': min(1.0, abs(sel_coef.s) * 100 + abs(alphagenome_effect))
    }
