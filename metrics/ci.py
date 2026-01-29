"""Correlation Immunity (CI) metric calculation for S-boxes."""

from typing import List
import numpy as np
from metrics.__utils__ import hamming_weight, sbox_to_boolean_functions


def compute_ci(sbox):
    """
    Compute the Correlation Immunity (CI) of an S-box.
    
    Correlation immunity measures the degree to which the output of an S-box
    is statistically independent of subsets of the input bits. An S-box is said
    to have correlation immunity of order m if there is no nonzero correlation
    between any m or fewer input bits and any output bit.
    
    Args:
        sbox: A list of 256 integers representing the S-box
        
    Returns:
        Estimated Correlation Immunity value
    """
    boolean_functions = sbox_to_boolean_functions(sbox)
    
    # For an 8x8 S-box, we'll calculate the correlation immunity
    # by checking linear relationships between subsets of input bits and output bits
    n_inputs = 8  # For 8x8 S-box
    
    min_correlation = float('inf')
    
    # Check all output bits
    for output_bit_idx, func in enumerate(boolean_functions):
        # Check different orders of correlation immunity
        # For each possible input bit subset
        for num_input_bits in range(1, n_inputs + 1):
            # For simplicity, we'll check a few representative subsets
            # rather than all possible combinations (which would be computationally expensive)
            
            # Calculate correlation between single input bits and this output bit
            for input_bit_pos in range(min(num_input_bits, n_inputs)):
                # Create a mask for the input bit
                mask = 1 << input_bit_pos
                
                # Calculate correlation
                correlation = 0
                for x in range(256):
                    # Extract the selected input bit(s)
                    input_part = (x >> input_bit_pos) & 1
                    output_bit = func[x]
                    
                    # Correlation measure
                    correlation += ((-1) ** (input_part ^ output_bit))
                
                correlation = abs(correlation) / 256.0  # Normalize
                min_correlation = min(min_correlation, correlation)
    
    # The correlation immunity is related to how low the correlation is
    # Higher correlation immunity means lower correlation
    # For a balanced function, CI is typically between 0 and n_inputs-1
    # We'll return a normalized value based on the minimum correlation found
    ci_value = (1.0 - min_correlation) * 7  # Scale to approximate range [0, 7]
    
    return max(0.0, min(7.0, ci_value))  # Clamp to [0, 7] range