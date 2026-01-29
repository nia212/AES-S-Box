"""
Nonlinearity metrics module.
"""

import numpy as np
from typing import Dict


def compute_boolean_function_nonlinearity(bit_vector: np.ndarray) -> int:
    """
    Compute nonlinearity of a Boolean function.
    
    Nonlinearity is the minimum Hamming distance to the set of affine Boolean functions.
    
    Args:
        bit_vector: Boolean function output (256 values, 0 or 1)
    
    Returns:
        Nonlinearity value (0-128)
    """
    if len(bit_vector) != 256:
        raise ValueError("Bit vector must have 256 elements")
    
    min_distance = 128
    
    # Check distance to all 512 affine functions (256 linear + 256 affine with constant)
    for linear_coeff in range(256):
        for constant in range(2):
            distance = 0
            
            for x in range(256):
                # Compute linear function <linear_coeff, x>
                linear_val = 0
                for bit_pos in range(8):
                    linear_val ^= ((linear_coeff >> bit_pos) & 1) * ((x >> bit_pos) & 1)
                
                affine_val = linear_val ^ constant
                
                if bit_vector[x] != affine_val:
                    distance += 1
            
            min_distance = min(min_distance, distance)
    
    return min_distance


def sbox_nonlinearity(sbox: np.ndarray) -> Dict:
    """
    Calculate nonlinearity for each bit of S-box.
    
    Args:
        sbox: S-box array
    
    Returns:
        Dictionary with nonlinearity metrics
    """
    nonlinearities = []
    details = {}
    
    for bit_pos in range(8):
        # Extract this bit from all S-box outputs
        bit_vector = np.array([(sbox[i] >> bit_pos) & 1 for i in range(256)], dtype=int)
        
        nl = compute_boolean_function_nonlinearity(bit_vector)
        nonlinearities.append(nl)
        details[f'bit_{bit_pos}'] = nl
    
    return {
        'nonlinearities': nonlinearities,
        'bit_details': details,
        'avg_nonlinearity': float(np.mean(nonlinearities)),
        
    }
