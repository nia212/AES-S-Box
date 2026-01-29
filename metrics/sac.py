"""
Strict Avalanche Criterion (SAC) metrics.
"""

import numpy as np
from typing import Dict


def compute_sac_matrix(sbox: np.ndarray) -> np.ndarray:
    """
    Compute SAC matrix for S-box.
    
    SAC matrix[i,j] = probability that output bit j flips when input bit i is flipped.
    
    Args:
        sbox: S-box array
    
    Returns:
        8x8 SAC matrix (probabilities)
    """
    sac_matrix = np.zeros((8, 8))
    
    for input_bit in range(8):
        for output_bit in range(8):
            flip_count = 0
            
            for value in range(256):
                flipped_value = value ^ (1 << input_bit)
                
                # Count output bit flips
                original_output_bit = (sbox[value] >> output_bit) & 1
                flipped_output_bit = (sbox[flipped_value] >> output_bit) & 1
                
                if original_output_bit != flipped_output_bit:
                    flip_count += 1
            
            sac_matrix[input_bit, output_bit] = flip_count / 256
    
    return sac_matrix


def sac_score(sbox: np.ndarray, threshold: float = 0.05) -> Dict:
    sac_matrix = compute_sac_matrix(sbox)
    
    # Count deviations from 0.5
    deviations = np.abs(sac_matrix - 0.5)
    max_deviation = np.max(deviations)
    mean_deviation = np.mean(deviations)
    
    # Count how many values exceed threshold
    violations = np.sum(deviations > threshold)
    
    return {
        'sac_matrix': sac_matrix,
        'max_deviation': float(max_deviation),
        'mean_deviation': float(mean_deviation),
        'violations': int(violations),
        'total_elements': 64,
        'violation_percentage': float(violations / 64 * 100),
        'passes_sac': violations == 0
    }
