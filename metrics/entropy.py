"""
Entropy metrics module for cryptographic analysis.
"""

import numpy as np
from typing import Dict


def shannon_entropy(data: np.ndarray) -> float:
    """
    Calculate Shannon entropy of data.
    
    H(X) = -Σ p(x_i) * log2(p(x_i))
    
    Args:
        data: Data array (values 0-255)
    
    Returns:
        Shannon entropy value (0-8 for bytes)
    """
    if len(data) == 0:
        return 0.0
    
    # Get histogram
    histogram, _ = np.histogram(data, bins=256, range=(0, 256))
    
    # Normalize
    probabilities = histogram / len(data)
    
    # Calculate entropy
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def entropy_analysis(plaintext: np.ndarray, ciphertext: np.ndarray) -> Dict[str, float]:
    """
    Analyze entropy of plaintext vs ciphertext.
    
    Args:
        plaintext: Plaintext data
        ciphertext: Encrypted data
    
    Returns:
        Dictionary with entropy metrics
    """
    plaintext_flat = plaintext.flatten() if isinstance(plaintext, np.ndarray) else np.frombuffer(plaintext, dtype=np.uint8)
    ciphertext_flat = ciphertext.flatten() if isinstance(ciphertext, np.ndarray) else np.frombuffer(ciphertext, dtype=np.uint8)
    
    plaintext_entropy = shannon_entropy(plaintext_flat)
    ciphertext_entropy = shannon_entropy(ciphertext_flat)
    
    return {
        'plaintext_entropy': plaintext_entropy,
        'ciphertext_entropy': ciphertext_entropy,
        'entropy_increase': ciphertext_entropy - plaintext_entropy,
        'ideal_entropy': 8.0
    }


def entropy_distribution(data: np.ndarray, bins: int = 256) -> Dict:
    """
    Get entropy distribution information.
    
    Args:
        data: Data array
        bins: Number of histogram bins
    
    Returns:
        Dictionary with distribution info
    """
    flat = data.flatten() if isinstance(data, np.ndarray) else np.frombuffer(data, dtype=np.uint8)
    
    histogram, _ = np.histogram(flat, bins=bins, range=(0, 256))
    
    return {
        'histogram': histogram.tolist(),
        'min_frequency': int(np.min(histogram)),
        'max_frequency': int(np.max(histogram)),
        'mean_frequency': float(np.mean(histogram)),
        'std_frequency': float(np.std(histogram))
    }
