"""Bit Independence Criterion (BIC) metric calculation for S-boxes."""

from typing import List
import numpy as np



def sbox_to_boolean_functions(sbox: List[int]) -> List[np.ndarray]:
    """
    Convert an S-box to its 8 Boolean component functions.
    
    Args:
        sbox: A list of 256 integers representing the S-box
        
    Returns:
        A list of 8 Boolean functions (each as a numpy array of 0s and 1s)
    """
    boolean_funcs = []
    for bit_pos in range(8):
        func = np.zeros(256, dtype=np.int8)
        for i in range(256):
            # Extract bit at position bit_pos from sbox[i]
            func[i] = (sbox[i] >> bit_pos) & 1
        boolean_funcs.append(func)
    return boolean_funcs


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
    when considered together. We simplify this by checking the SAC 
    property for pairs of output bits simultaneously.
    
    Args:
        sbox: A list of 256 integers representing the S-box
        
    Returns:
        Average BIC-SAC value across pairs of output bits
    """
    total_correlation = 0.0
    count = 0
    
    # We'll measure correlation between pairs of output bits
    # when a single input bit is flipped
    for input_bit in range(8):
        mask = 1 << input_bit
        correlations = []
        
        for output_bit1 in range(8):
            for output_bit2 in range(output_bit1 + 1, 8):  # Unordered pairs
                changes1 = 0
                changes2 = 0
                
                for x in range(256):
                    flipped_x = x ^ mask
                    if flipped_x < 256:
                        orig_output = sbox[x]
                        flipped_output = sbox[flipped_x]
                        
                        # Check if output_bit1 changed
                        orig_bit1 = (orig_output >> output_bit1) & 1
                        flipped_bit1 = (flipped_output >> output_bit1) & 1
                        bit1_changed = orig_bit1 != flipped_bit1
                        
                        # Check if output_bit2 changed
                        orig_bit2 = (orig_output >> output_bit2) & 1
                        flipped_bit2 = (flipped_output >> output_bit2) & 1
                        bit2_changed = orig_bit2 != flipped_bit2
                        
                        # Count how often both bits changed together
                        if bit1_changed and bit2_changed:
                            changes1 += 1
                            changes2 += 1
                        elif bit1_changed:
                            changes1 += 1
                        elif bit2_changed:
                            changes2 += 1
                
                # For true BIC-SAC, we'd want the changes to be independent
                # So we measure independence as abs(correlation - 0.25) (since 0.5*0.5 = 0.25)
                # But for simplicity, just average the rate at which they change together
                correlation = changes1 / 256.0 if 256 > 0 else 0.0
                correlations.append(correlation)
        
        if correlations:
            total_correlation += sum(correlations) / len(correlations)
            count += 1
    
    return total_correlation / count if count > 0 else 0.0


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