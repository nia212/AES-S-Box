"""
Nonlinearity metrics module.
"""

import numpy as np
from typing import Dict


def compute_boolean_function_nonlinearity(bit_vector: np.ndarray) -> int:
    """
    Compute nonlinearity of a Boolean function using Walsh-Hadamard transform.
    
    Nonlinearity is the minimum Hamming distance to the set of affine Boolean functions.
    
    Args:
        bit_vector: Boolean function output (256 values, 0 or 1)
    
    Returns:
        Nonlinearity value (0-128)
    """
    if len(bit_vector) != 256:
        raise ValueError("Bit vector must have 256 elements")
    
    # Convert to {-1, +1} representation for Walsh transform
    f_plus_minus = (-1.0) ** bit_vector.astype(float)
    
    # Apply Fast Walsh-Hadamard Transform
    W = f_plus_minus.copy()
    h = 1
    while h < 256:
        for i in range(0, 256, h * 2):
            for j in range(i, i + h):
                u = W[j]
                v = W[j + h]
                W[j] = u + v
                W[j + h] = u - v
        h *= 2
    
    # Nonlinearity = 128 - max(|W(a)|)/2
    max_walsh = np.max(np.abs(W))
    nl = 128 - int(np.ceil(max_walsh / 2))
    
    return nl


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
