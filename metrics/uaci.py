"""
UACI and combined image metrics module.
"""

import numpy as np
from typing import Dict


def uaci_extended(plaintext_img: np.ndarray, ciphertext_img: np.ndarray) -> Dict:
    """
    Extended UACI analysis with additional metrics.
    
    Args:
        plaintext_img: Original image
        ciphertext_img: Encrypted image
    
    Returns:
        Dictionary with comprehensive UACI metrics
    """
    if plaintext_img.shape != ciphertext_img.shape:
        raise ValueError("Images must have same shape")
    
    plaintext_float = plaintext_img.astype(float)
    ciphertext_float = ciphertext_img.astype(float)
    
    diff = np.abs(plaintext_float - ciphertext_float)
    total_pixels = plaintext_img.size
    
    # UACI calculation
    uaci_value = (np.sum(diff) / (total_pixels * 255)) * 100
    
    # Additional metrics
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    min_diff = np.min(diff)
    max_diff = np.max(diff)
    
    # Expected UACI for random encryption
    expected_uaci = 33.4635
    
    return {
        'uaci': uaci_value,
        'expected_uaci': expected_uaci,
        'deviation': abs(uaci_value - expected_uaci),
        'mean_pixel_difference': float(mean_diff),
        'std_pixel_difference': float(std_diff),
        'min_pixel_difference': float(min_diff),
        'max_pixel_difference': float(max_diff),
        'is_good_uaci': 30.0 < uaci_value < 35.0
    }


def image_quality_metrics(original_img: np.ndarray, encrypted_img: np.ndarray) -> Dict:
    """
    Calculate image quality degradation metrics.
    
    Args:
        original_img: Original image
        encrypted_img: Encrypted image
    
    Returns:
        Dictionary with quality metrics
    """
    if original_img.shape != encrypted_img.shape:
        raise ValueError("Images must have same shape")
    
    original_float = original_img.astype(float)
    encrypted_float = encrypted_img.astype(float)
    
    # Mean Squared Error (MSE)
    mse = np.mean((original_float - encrypted_float) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = 255.0
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
    
    # Correlation
    original_mean = np.mean(original_float)
    encrypted_mean = np.mean(encrypted_float)
    
    cov = np.mean((original_float - original_mean) * (encrypted_float - encrypted_mean))
    std_orig = np.std(original_float)
    std_enc = np.std(encrypted_float)
    
    correlation = cov / (std_orig * std_enc) if (std_orig * std_enc) > 0 else 0.0
    
    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'correlation': float(correlation),
        'is_good_encryption': psnr < 10.0  # Low PSNR means good encryption
    }


def histogram_metrics(plaintext_img: np.ndarray, ciphertext_img: np.ndarray, 
                     bins: int = 256) -> Dict:
    """
    Analyze histogram distribution changes.
    
    Args:
        plaintext_img: Original image
        ciphertext_img: Encrypted image
        bins: Number of histogram bins
    
    Returns:
        Dictionary with histogram metrics
    """
    plain_flat = plaintext_img.flatten()
    cipher_flat = ciphertext_img.flatten()
    
    # Compute histograms
    plain_hist, _ = np.histogram(plain_flat, bins=bins, range=(0, 256))
    cipher_hist, _ = np.histogram(cipher_flat, bins=bins, range=(0, 256))
    
    # Normalize
    plain_hist_norm = plain_hist / len(plain_flat)
    cipher_hist_norm = cipher_hist / len(cipher_flat)
    
    # Chi-square test
    chi_square = 0.0
    for i in range(bins):
        expected = 1.0 / bins  # Uniform distribution
        if cipher_hist_norm[i] > 0:
            chi_square += ((cipher_hist_norm[i] - expected) ** 2) / expected
    
    # Entropy of ciphertext
    ciphertext_entropy = 0.0
    for count in cipher_hist:
        if count > 0:
            p = count / len(cipher_flat)
            ciphertext_entropy -= p * np.log2(p)
    
    # Expected entropy for uniform distribution
    expected_entropy = np.log2(bins)
    
    return {
        'plaintext_histogram': plain_hist.tolist(),
        'ciphertext_histogram': cipher_hist.tolist(),
        'chi_square': float(chi_square),
        'ciphertext_entropy': float(ciphertext_entropy),
        'expected_entropy': float(expected_entropy),
        'entropy_diff': float(abs(ciphertext_entropy - expected_entropy)),
        'is_good_histogram': ciphertext_entropy > 7.5  # > 7.5 for 256-level
    }
