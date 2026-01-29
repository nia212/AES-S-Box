"""
Differential Uniformity (DU) metrics module.
"""

import numpy as np
from typing import Dict
from collections import defaultdict


def compute_ddt(sbox: np.ndarray) -> np.ndarray:
    """
    Compute Differential Distribution Table (DDT).
    
    DDT[a, b] = number of x such that S(x) XOR S(x XOR a) = b
    
    Args:
        sbox: S-box array
    
    Returns:
        256x256 DDT matrix
    """
    ddt = np.zeros((256, 256), dtype=int)
    
    for x in range(256):
        for delta_in in range(256):
            delta_out = sbox[x] ^ sbox[(x ^ delta_in) & 0xff]
            ddt[delta_in, delta_out] += 1
    
    return ddt


def differential_uniformity(sbox: np.ndarray) -> Dict:
    """
    Calculate differential uniformity of S-box.
    
    Args:
        sbox: S-box array
    
    Returns:
        Dictionary with DU metrics
    """
    ddt = compute_ddt(sbox)
    
    # Exclude DDT[0, 0] which is always 256
    # Maximum value should be as close to 1 as possible
    ddt_no_zero = ddt[1:, :]  # Exclude delta_in = 0
    max_du = np.max(ddt_no_zero)
    min_du = np.min(ddt_no_zero[ddt_no_zero > 0])  # Minimum non-zero
    
    # Count distribution
    distribution = defaultdict(int)
    for i in range(1, 256):
        for j in range(256):
            count = ddt[i, j]
            if count > 0:
                distribution[count] += 1
    
    return {
        'ddt': ddt,
        'max_differential_count': int(max_du),
        'min_differential_count': int(min_du),
        'distribution': dict(distribution),
        'is_perfect': max_du == 1,  # Perfect if all non-zero entries are 1
        'uniformity_score': float(1.0 / max_du) if max_du > 0 else 0.0
    }


def linear_approximation_probability(sbox: np.ndarray) -> Dict:
    """
    Calculate Linear Approximation Probability (LAP) metrics.
    
    Args:
        sbox: S-box array
    
    Returns:
        Dictionary with LAP metrics
    """
    # Compute Linear Approximation Table (LAT)
    lat = np.zeros((256, 256), dtype=int)
    
    for input_mask in range(256):
        for output_mask in range(256):
            count = 0
            
            for x in range(256):
                # Compute input parity <input_mask, x>
                in_parity = 0
                for bit_pos in range(8):
                    in_parity ^= ((input_mask >> bit_pos) & 1) * ((x >> bit_pos) & 1)
                
                # Compute output parity <output_mask, S(x)>
                out_parity = 0
                for bit_pos in range(8):
                    out_parity ^= ((output_mask >> bit_pos) & 1) * ((sbox[x] >> bit_pos) & 1)
                
                if in_parity == out_parity:
                    count += 1
            
            lat[input_mask, output_mask] = abs(count - 128)
    
    max_lap = np.max(lat)
    min_lap = np.min(lat[lat > 0])
    
    # Bias is LAT value / 256
    max_bias = max_lap / 256
    
    return {
        'lat': lat,
        'max_lap': int(max_lap),
        'min_lap': int(min_lap),
        'max_bias': float(max_bias),
        'min_bias': float(min_lap / 256),
        'is_resistant': max_lap < 128  # < 128 means good resistance
    }
