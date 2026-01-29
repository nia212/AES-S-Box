"""Bit Independence Criterion (BIC) metric calculation for S-boxes."""

from typing import List
import numpy as np
from metrics.__utils__ import sbox_to_boolean_functions


def compute_bic_nl(sbox):
    """
    Compute the Bit Independence Criterion - Nonlinearity (BIC-NL) of an S-box.
    
    BIC-NL measures the nonlinearity of XOR sums of multiple output bits.
    For computational simplicity, we calculate the average nonlinearity 
    of XOR combinations of all pairs of output bits.
    
    Args:
        sbox: A list of 256 integers representing the S-box
        
    Returns:
        Average BIC-NL value across pairs of output bits
    """
    boolean_functions = sbox_to_boolean_functions(sbox)
    n = len(boolean_functions)  # Should be 8 for an 8x8 S-box
    bic_values = []
    
    # Calculate nonlinearity for XOR combinations of every pair of output bits
    for i in range(n):
        for j in range(i + 1, n):
            # XOR the two Boolean functions
            combined_func = np.bitwise_xor(boolean_functions[i], boolean_functions[j])
            
            # Calculate nonlinearity of the combined function
            # We need to temporarily use a function similar to compute_nl but for single function
            walsh_spectrum = _walsh_transform_single(combined_func)
            max_walsh = np.max(np.abs(walsh_spectrum))
            nl_combined = 128 - max_walsh / 2  # For 8-bit input
            bic_values.append(int(round(nl_combined)))
    
    # Return the average BIC-NL value
    return sum(bic_values) / len(bic_values) if bic_values else 0.0


def compute_bic_sac(sbox):
    """
    Compute the Bit Independence Criterion - SAC (BIC-SAC) of an S-box.
    
    BIC-SAC measures how well the output bits satisfy the SAC property
    when considered together using vectorized operations.
    
    Args:
        sbox: A list of 256 integers representing the S-box
        
    Returns:
        Average BIC-SAC value across pairs of output bits
    """
    sbox = np.array(sbox, dtype=np.uint8)
    bic_sac_values = []
    
    # For each input bit position
    for input_bit in range(8):
        mask = 1 << input_bit
        
        # Create all flipped indices
        flipped_indices = np.arange(256) ^ mask
        
        # Get original and flipped outputs (vectorized)
        orig_outputs = sbox
        flipped_outputs = sbox[flipped_indices]
        
        # Compute changes for each bit pair
        for bit1 in range(8):
            for bit2 in range(bit1 + 1, 8):
                # Extract bits (vectorized)
                orig_bit1 = (orig_outputs >> bit1) & 1
                flipped_bit1 = (flipped_outputs >> bit1) & 1
                orig_bit2 = (orig_outputs >> bit2) & 1
                flipped_bit2 = (flipped_outputs >> bit2) & 1
                
                # Count changes
                changes1 = np.sum(orig_bit1 != flipped_bit1)
                changes2 = np.sum(orig_bit2 != flipped_bit2)
                
                # SAC should be close to 0.5
                sac1 = changes1 / 256.0
                sac2 = changes2 / 256.0
                
                # Measure independence (should be close to 0.5)
                independence = abs(0.5 - abs(sac1 - sac2))
                bic_sac_values.append(independence)
    
    return np.mean(bic_sac_values) * 100 if bic_sac_values else 50.0


def _walsh_transform_single(func):
    """
    Helper function to compute Walsh-Hadamard transform of a single Boolean function.
    """
    # Convert to {-1, +1} representation
    f_plus_minus = (-1) ** func
    
    # Apply Walsh-Hadamard transform using recursive algorithm
    n = len(f_plus_minus)
    W = f_plus_minus.copy().astype(np.float64)
    
    # Iteratively apply Hadamard matrix
    h = 2
    while h <= n:
        for i in range(0, n, h):
            for j in range(h // 2):
                u = W[i + j]
                v = W[i + j + h // 2]
                W[i + j] = u + v
                W[i + j + h // 2] = u - v
        h *= 2
    
    return W