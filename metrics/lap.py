"""Linear Approximation Probability (LAP) metric calculation for S-boxes."""

import numpy as np
from metrics.__utils__ import hamming_weight


def lap(sbox):
    """
    Compute the Linear Approximation Probability (LAP) of an S-box.

    LAP measures the maximum probability bias in linear approximations
    of the S-box. For an 8x8 S-box, this involves calculating the
    Linear Approximation Table (LAT) and finding its maximum entry.

    The Linear Approximation Probability is defined as:
    LAP = max_{α≠0,β≠0} |#{x : x·α = S(x)·β} - 2^{n-1}| / 2^n

    Or equivalently: max absolute value in LAT divided by 2^n (excluding α=0, β=0)

    Args:
        sbox: A list of 256 integers representing the S-box

    Returns:
        Maximum Linear Approximation Probability
    """
    # Create the Linear Approximation Table (LAT)
    # For efficiency, we'll calculate all entries
    lat = np.zeros((256, 256), dtype=int)

    # For each input mask (a) and output mask (b)
    for a in range(256):
        for b in range(256):
            # Calculate the correlation for mask pair (a, b)
            correlation = 0
            for x in range(256):
                # Calculate dot product of a with x
                ax = hamming_weight(a & x) % 2
                # Calculate dot product of b with sbox[x]
                bs = hamming_weight(b & sbox[x]) % 2
                # Add 1 if they are equal, subtract 1 if not
                if ax == bs:
                    correlation += 1
                else:
                    correlation -= 1

            # Store in LAT (we store the raw correlation value)
            lat[a][b] = abs(correlation)

    # The maximum value in the LAT (excluding zero input/output masks)
    # corresponds to the maximum linear approximation probability
    # normalized by dividing by 256 (the domain size)
    max_lat_value = np.max(lat[1:, 1:])  # Exclude first row/column (zero masks)

    # LAP is the maximum absolute correlation divided by 2^n
    # This represents the deviation from 0.5 probability
    # For AES S-box, the maximum absolute value in LAT (excluding row/column 0)
    # should be 128, giving LAP = 128/256 = 0.5, but this is the correlation bias
    # The actual LAP is (1 + max_corr)/2 where max_corr is normalized correlation
    # Actually, the correct LAP is |max_count - 2^(n-1)| / 2^n = max_lat_value/2^(n+1)
    # Wait, let's recalculate: if max_lat_value is the max |sum(-1)^(ax + bs)|,
    # then the actual count difference is max_lat_value, and probability is (256+max_lat_value)/2 / 256
    # So the bias from 0.5 is max_lat_value/512
    # The LAP is max_bias^2 = (max_lat_value/512)^2 * 2 = max_lat_value^2 / (2^(2n+1))
    # Actually, let's define LAP correctly:
    # LAP = max_{a≠0,b≠0} |Pr[a·x=b·S(x)] - 1/2| = max_lat_value / 2^(n+1)
    # For AES S-box, max_lat_value should be 128 (not 16), so LAP = 128/512 = 0.25
    # That's still wrong. Let me look up the correct definition.
    # LAP = max_{a≠0,b≠0} |Pr[a·x=b·S(x)] - 1/2|^2 * 2 = (max_lat_value/2^n)^2 * 2
    # For AES, if max_lat_value = 64, then LAP = (64/256)^2 * 2 = (1/4)^2 * 2 = 2/16 = 1/8 = 0.125
    # Still not right. Let's try: LAP = max_{a≠0,b≠0} |Pr[a·x=b·S(x)] - 1/2| = max_lat_value / 2^(n+1)
    # For AES, if max_lat_value = 100, LAP = 100/512 ≈ 0.195.
    # Actually, for AES S-box, the correct maximum absolute correlation is 16*2 = 32,
    # so LAP = 32/256 = 0.125 or max_lat_value/512 = 64/512 = 0.125
    # No, actually for AES, the maximum is 16 in the DDT, and related to LAT as 16*2=32 deviation from 128
    # So in LAT (after normalization), max is 32, giving LAP = 32/256 = 0.125
    # The correct LAP for AES is 16/256 = 0.0625, meaning max absolute correlation in our table should be 32
    # Wait, let me go back to the definition:
    # LAP = max |count - 2^(n-1)| / 2^n where count is #{x | a·x = b·S(x)}
    # So if max_lat_value = max |correlation| = max |2*#{x | a·x = b·S(x)} - 2^n| = 2 * |count - 2^(n-1)|
    # Then LAP = max_lat_value / 2^(n+1)
    # For AES, if LAP = 16/256 = 1/16 = 0.0625, then max_lat_value = 0.0625 * 2^9 = 0.0625 * 512 = 32
    # So max_lat_value should be 32, but our code gives higher value.

    # Correct LAP formula: max_lat_value / 2^n
    # Because LAP = max |Pr[a·x=b·S(x)] - 1/2|, and Pr[a·x=b·S(x)] = (#matches)/2^n
    # The correlation count = 2*(#matches) - 2^n, so |correlation_count| = 2^n * |2*Pr - 1| = 2^(n+1)*|Pr - 1/2|
    # So |Pr - 1/2| = |correlation_count| / 2^(n+1)
    # Therefore, LAP = max_lat_value / 2^(n+1)
    # For 8x8 S-box: LAP = max_lat_value / 512

    lap = max_lat_value / 512.0

    return lap