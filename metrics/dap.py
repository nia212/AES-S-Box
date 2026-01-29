"""Differential Approximation Probability (DAP) metric calculation for S-boxes."""


def dap(sbox):
    """
    Compute the Differential Approximation Probability (DAP) of an S-box.
    
    DAP measures the maximum probability of differential characteristics
    in the S-box. This is calculated using the Difference Distribution Table (DDT).
    
    Args:
        sbox: A list of 256 integers representing the S-box
        
    Returns:
        Maximum Differential Approximation Probability
    """
    # Create the Difference Distribution Table (DDT)
    ddt = [[0 for _ in range(256)] for _ in range(256)]
    
    # For each possible input difference dx
    for dx in range(1, 256):  # Skip dx=0 as it's not interesting
        # For each possible input x
        for x in range(256):
            # Calculate corresponding input x'
            x_prime = x ^ dx
            # Calculate output difference dy = sbox[x] ^ sbox[x']
            dy = sbox[x] ^ sbox[x_prime]
            # Increment count in DDT
            ddt[dx][dy] += 1
    
    # Find the maximum value in the DDT (excluding the dx=0 row)
    max_dap = 0
    for dx in range(1, 256):
        for dy in range(256):
            if ddt[dx][dy] > max_dap:
                max_dap = ddt[dx][dy]
    
    # Normalize by dividing by 256 (total number of x values for each dx)
    normalized_dap = max_dap / 256.0
    
    return normalized_dap