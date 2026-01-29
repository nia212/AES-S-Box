"""Utility functions shared across metrics modules."""

import numpy as np
from typing import List


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


def hamming_weight(value: int) -> int:
    """
    Compute the Hamming weight (number of 1-bits) of an integer.
    
    Args:
        value: An integer value
        
    Returns:
        The number of 1-bits in the binary representation
    """
    count = 0
    while value:
        count += value & 1
        value >>= 1
    return count
