"""Algebraic Degree (AD) metric calculation for S-boxes."""

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


def compute_ad(sbox):
    """
    Compute the Algebraic Degree (AD) of an S-box.
    
    The algebraic degree of an S-box is the highest algebraic degree 
    among all of its component Boolean functions. The algebraic degree 
    of a Boolean function is the degree of its Algebraic Normal Form (ANF).
    
    Args:
        sbox: A list of 256 integers representing the S-box
        
    Returns:
        Maximum algebraic degree across all component functions
    """
    boolean_functions = sbox_to_boolean_functions(sbox)
    degrees = []
    
    for func in boolean_functions:
        # Calculate the algebraic degree for each component function
        degree = _calculate_algebraic_degree(func)
        degrees.append(degree)
    
    # Return the maximum degree
    return max(degrees) if degrees else 0


def _calculate_algebraic_degree(function):
    """
    Calculate the algebraic degree of a Boolean function.
    
    This uses the concept that the algebraic degree is the highest
    degree of any term in the Algebraic Normal Form (ANF).
    For an n-variable Boolean function, we can determine the degree
    by checking the highest order terms in the ANF.
    """
    # For an 8-variable Boolean function, the maximum possible degree is 8
    n = 8  # Number of variables (input bits for 8x8 S-box)
    
    # Convert the truth table to ANF coefficients using the Möbius transform
    # This is essentially computing the algebraic normal form coefficients
    
    # Start with the truth table values
    anf = function.astype(int).copy()
    
    # Perform the ANF transformation (Möbius transform)
    for var in range(n):
        # Process all values that differ in the current variable
        for idx in range(256):
            if (idx >> var) & 1:  # Check if the variable-th bit is set
                anf[idx] ^= anf[idx ^ (1 << var)]  # XOR with value that differs only in variable-th bit
    
    # Find the highest degree terms in the ANF
    max_degree = 0
    for idx in range(256):
        # The degree of the term is the hamming weight of its index
        # Each bit position represents a variable in the monomial
        term_degree = bin(idx).count('1')
        if anf[idx] & 1:  # If the coefficient is odd (non-zero in GF(2))
            max_degree = max(max_degree, term_degree)
    
    return max_degree