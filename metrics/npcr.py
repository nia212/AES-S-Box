"""
NPCR and UACI image cryptography metrics.
"""

import numpy as np
from typing import Dict, Tuple


def npcr(plaintext_img: np.ndarray, ciphertext_img: np.ndarray) -> float:
    """
    Calculate Number of Pixels Change Rate (NPCR).
    
    NPCR = (number of different pixels / total pixels) * 100%
    
    Args:
        plaintext_img: Original image
        ciphertext_img: Encrypted image
    
    Returns:
        NPCR percentage (0-100)
    """
    if plaintext_img.shape != ciphertext_img.shape:
        raise ValueError("Images must have same shape")
    
    different_pixels = np.sum(plaintext_img != ciphertext_img)
    total_pixels = plaintext_img.size
    
    npcr_value = (different_pixels / total_pixels) * 100
    return npcr_value


def uaci(plaintext_img: np.ndarray, ciphertext_img: np.ndarray) -> float:
    """
    Calculate Unified Average Changing Intensity (UACI).
    
    UACI = (1 / (W*H*255)) * Σ |P(i,j) - C(i,j)| * 100%
    
    Args:
        plaintext_img: Original image
        ciphertext_img: Encrypted image
    
    Returns:
        UACI percentage (0-100)
    """
    if plaintext_img.shape != ciphertext_img.shape:
        raise ValueError("Images must have same shape")
    
    plaintext_float = plaintext_img.astype(float)
    ciphertext_float = ciphertext_img.astype(float)
    
    diff_sum = np.sum(np.abs(plaintext_float - ciphertext_float))
    total_pixels = plaintext_img.size
    
    uaci_value = (diff_sum / (total_pixels * 255)) * 100
    return uaci_value


def npcr_uaci_analysis(plaintext_img: np.ndarray, ciphertext_img: np.ndarray) -> Dict[str, float]:
    """
    Perform comprehensive NPCR/UACI analysis.
    
    Args:
        plaintext_img: Original image
        ciphertext_img: Encrypted image
    
    Returns:
        Dictionary with NPCR/UACI metrics
    """
    npcr_value = npcr(plaintext_img, ciphertext_img)
    uaci_value = uaci(plaintext_img, ciphertext_img)
    
    # Expected values for random encryption
    expected_npcr = 99.6094  # For 8-bit random images
    expected_uaci = 33.4635  # For 8-bit random images
    
    return {
        'npcr': npcr_value,
        'uaci': uaci_value,
        'expected_npcr': expected_npcr,
        'expected_uaci': expected_uaci,
        'npcr_diff': abs(npcr_value - expected_npcr),
        'uaci_diff': abs(uaci_value - expected_uaci),
        'is_good_npcr': npcr_value > 99.0,
        'is_good_uaci': 30.0 < uaci_value < 35.0,
        'is_good_encryption': (npcr_value > 99.0) and (30.0 < uaci_value < 35.0)
    }


def differential_analysis(plaintext_img: np.ndarray, ciphertext_img1: np.ndarray, 
                         ciphertext_img2: np.ndarray) -> Dict[str, float]:
    """
    Analyze differential attack resistance using two ciphertexts.
    
    Args:
        plaintext_img: Original image
        ciphertext_img1: First encryption
        ciphertext_img2: Second encryption (with one pixel changed in plaintext)
    
    Returns:
        Dictionary with differential analysis metrics
    """
    npcr_val = npcr(ciphertext_img1, ciphertext_img2)
    uaci_val = uaci(ciphertext_img1, ciphertext_img2)
    
    return {
        'differential_npcr': npcr_val,
        'differential_uaci': uaci_val,
        'is_resistant': (npcr_val > 99.0) and (30.0 < uaci_val < 35.0)
    }


def channel_wise_npcr_uaci(plaintext_rgb: np.ndarray, ciphertext_rgb: np.ndarray) -> Dict:
    """
    Calculate NPCR/UACI for each RGB channel separately.
    
    Args:
        plaintext_rgb: RGB plaintext image (H x W x 3)
        ciphertext_rgb: RGB ciphertext image (H x W x 3)
    
    Returns:
        Dictionary with per-channel metrics
    """
    if plaintext_rgb.shape != ciphertext_rgb.shape or plaintext_rgb.shape[2] != 3:
        raise ValueError("Images must be RGB (H x W x 3)")
    
    channels = ['Red', 'Green', 'Blue']
    results = {}
    
    for ch_idx, ch_name in enumerate(channels):
        plain_ch = plaintext_rgb[:, :, ch_idx]
        cipher_ch = ciphertext_rgb[:, :, ch_idx]
        
        npcr_ch = npcr(plain_ch, cipher_ch)
        uaci_ch = uaci(plain_ch, cipher_ch)
        
        results[ch_name] = {
            'npcr': npcr_ch,
            'uaci': uaci_ch
        }
    
    # Overall metrics
    results['Overall'] = npcr_uaci_analysis(plaintext_rgb, ciphertext_rgb)
    
    return results
