
import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations
from collections import defaultdict


class SBoxValidator:
    """Validate cryptographic properties of S-boxes."""
    
    @staticmethod
    def validate_bijectivity(sbox: np.ndarray) -> Tuple[bool, str]:
        """
        Check if S-box is bijective (permutation property).
        
        Args:
            sbox: S-box array (should be 256 entries for 8-bit)
        
        Returns:
            Tuple of (is_bijective, message)
        """
        if len(sbox) != 256:
            return False, f"S-box length must be 256, got {len(sbox)}"
        
        unique_values = np.unique(sbox)
        if len(unique_values) != 256:
            return False, f"S-box is not bijective: only {len(unique_values)} unique values"
        
        return True, "S-box is bijective"
    
    @staticmethod
    def validate_balance(sbox: np.ndarray, threshold: float = 0.05) -> Tuple[bool, str, Dict]:
        """
        Check balanced property: each output bit should have roughly equal probability.
        
        Args:
            sbox: S-box array
            threshold: Acceptable deviation from 0.5 (0-1)
        
        Returns:
            Tuple of (is_balanced, message, details)
        """
        if len(sbox) != 256:
            return False, "Invalid S-box length", {}
        
        details = {}
        all_balanced = True
        
        # Check each output bit
        for bit_pos in range(8):
            bit_count = sum(1 for val in sbox if (val >> bit_pos) & 1)
            balance = bit_count / 256
            details[f'bit_{bit_pos}'] = balance
            
            if abs(balance - 0.5) > threshold:
                all_balanced = False
        
        if all_balanced:
            return True, "S-box is balanced", details
        else:
            return False, "S-box is not balanced", details
    
    @staticmethod
    def strict_avalanche_criterion(sbox: np.ndarray, threshold: float = 0.05) -> Tuple[bool, str, Dict]:
        """
        Strict Avalanche Criterion (SAC): Flipping any input bit should flip each output bit with 50% probability.
        
        Args:
            sbox: S-box array
            threshold: Acceptable deviation from 0.5
        
        Returns:
            Tuple of (passes_sac, message, details)
        """
        if len(sbox) != 256:
            return False, "Invalid S-box length", {}
        
        sac_values = []
        details = {}
        
        # For each input bit position
        for bit_pos in range(8):
            bit_flip_count = defaultdict(int)
            
            # For each value in S-box
            for i in range(256):
                flipped_i = i ^ (1 << bit_pos)
                if flipped_i < 256:
                    # XOR of outputs when input bit is flipped
                    output_diff = sbox[i] ^ sbox[flipped_i]
                    
                    # Count how many output bits flip
                    for out_bit in range(8):
                        if (output_diff >> out_bit) & 1:
                            bit_flip_count[out_bit] += 1
            
            # Calculate SAC for this input bit
            sac_input_bit = []
            for out_bit in range(8):
                flip_prob = bit_flip_count[out_bit] / 256
                sac_input_bit.append(flip_prob)
                details[f'in_bit_{bit_pos}_out_bit_{out_bit}'] = flip_prob
            
            sac_values.extend(sac_input_bit)
        
        # Check if all probabilities are close to 0.5
        sac_passes = all(abs(val - 0.5) <= threshold for val in sac_values)
        
        if sac_passes:
            return True, "S-box passes Strict Avalanche Criterion", details
        else:
            return False, "S-box fails Strict Avalanche Criterion", details
    
    @staticmethod
    def differential_uniformity(sbox: np.ndarray) -> Tuple[bool, str, Dict]:
        """
        Calculate differential uniformity: for all a != 0, the number of x such that S(x) XOR S(x XOR a) = b.
        
        Args:
            sbox: S-box array
        
        Returns:
            Tuple of (is_good, message, details)
        """
        if len(sbox) != 256:
            return False, "Invalid S-box length", {}
        
        # Compute differential distribution table
        ddt = np.zeros((256, 256), dtype=int)
        
        for x in range(256):
            for delta_in in range(1, 256):  # Skip delta_in = 0
                delta_out = sbox[x] ^ sbox[x ^ delta_in]
                ddt[delta_in, delta_out] += 1
        
        # Find maximum uniformity value (excluding DDT[0][0])
        max_uniformity = np.max(ddt[1:, :])  # Skip delta_in = 0
        
        details = {
            'max_differential_count': int(max_uniformity),
            'ideal_value': 1,
            'uniformity_score': max(0, 1.0 - (max_uniformity / 256))
        }
        
        # Good differential uniformity is close to 1
        is_good = max_uniformity <= 4  # Stricter than purely uniform
        
        if is_good:
            return True, f"Good differential uniformity (max count: {max_uniformity})", details
        else:
            return False, f"Poor differential uniformity (max count: {max_uniformity})", details
    
    @staticmethod
    def nonlinearity(sbox: np.ndarray) -> Tuple[bool, str, Dict]:
        """
        Calculate nonlinearity: minimum distance to affine functions.
        
        Args:
            sbox: S-box array
        
        Returns:
            Tuple of (is_good, message, details)
        """
        if len(sbox) != 256:
            return False, "Invalid S-box length", {}
        
        # For each output bit, compute its nonlinearity
        nl_values = []
        details = {}
        
        for bit_pos in range(8):
            # Extract this bit from all S-box outputs
            bit_vector = np.array([(sbox[i] >> bit_pos) & 1 for i in range(256)])
            
            # Compute nonlinearity of this Boolean function
            nl = SBoxValidator._compute_boolean_function_nonlinearity(bit_vector)
            nl_values.append(nl)
            details[f'bit_{bit_pos}_nonlinearity'] = nl
        
        avg_nl = np.mean(nl_values)
        min_nl = np.min(nl_values)
        details['average_nonlinearity'] = avg_nl
        details['minimum_nonlinearity'] = min_nl
        
        # Good nonlinearity should be > 100 for 8-bit functions
        is_good = min_nl > 100
        
        if is_good:
            return True, f"Good nonlinearity (min: {min_nl}, avg: {avg_nl:.2f})", details
        else:
            return False, f"Poor nonlinearity (min: {min_nl}, avg: {avg_nl:.2f})", details
    
    @staticmethod
    def _compute_boolean_function_nonlinearity(bit_vector: np.ndarray) -> float:
        """
        Compute nonlinearity of a Boolean function.
        
        Args:
            bit_vector: Boolean function output (256 values)
        
        Returns:
            Nonlinearity value
        """
        min_distance = 128  # Maximum possible distance
        
        # Check distance to all affine functions
        # For 8-bit functions, we need to check 256 affine functions
        # (2^8 possible coefficients for linear part + 1 for constant)
        
        for linear_coeff in range(256):
            for constant in range(2):
                distance = 0
                for x in range(256):
                    # Compute linear function: <linear_coeff, x> XOR constant
                    linear_val = 0
                    for bit_pos in range(8):
                        linear_val ^= ((linear_coeff >> bit_pos) & 1) * ((x >> bit_pos) & 1)
                    affine_val = linear_val ^ constant
                    
                    # Count disagreements
                    if bit_vector[x] != affine_val:
                        distance += 1
                
                # Update minimum distance
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    @staticmethod
    def linear_approximation_probability(sbox: np.ndarray) -> Tuple[bool, str, Dict]:
        """
        Compute Linear Approximation Probability (LAP) or Linear Bias.
        
        Args:
            sbox: S-box array
        
        Returns:
            Tuple of (is_good, message, details)
        """
        if len(sbox) != 256:
            return False, "Invalid S-box length", {}
        
        max_bias = 0.0
        details = {}
        
        # For each input and output mask
        for input_mask in range(1, 256):  # Skip 0
            for output_mask in range(256):  # Include 0
                count = 0
                for x in range(256):
                    # Compute input and output parities
                    in_parity = bin(x & input_mask).count('1') % 2
                    out_parity = bin(sbox[x] & output_mask).count('1') % 2
                    
                    if in_parity == out_parity:
                        count += 1
                
                # Bias is |count - 128| / 256
                bias = abs(count - 128) / 256
                if bias > max_bias:
                    max_bias = bias
        
        details['max_linear_bias'] = max_bias
        details['max_lap'] = 2 * max_bias
        
        # Good LAP should be < 0.1
        is_good = max_bias < 0.1
        
        if is_good:
            return True, f"Good linear approximation probability (max bias: {max_bias:.4f})", details
        else:
            return False, f"Poor linear approximation probability (max bias: {max_bias:.4f})", details
    
    @staticmethod
    def full_validation(sbox: np.ndarray) -> Dict[str, Tuple[bool, str, Dict]]:
        """
        Perform complete validation of S-box.
        
        Args:
            sbox: S-box array
        
        Returns:
            Dictionary of validation results
        """
        results = {
            'bijectivity': SBoxValidator.validate_bijectivity(sbox),
            'balance': SBoxValidator.validate_balance(sbox),
            'sac': SBoxValidator.strict_avalanche_criterion(sbox),
            'differential_uniformity': SBoxValidator.differential_uniformity(sbox),
            'nonlinearity': SBoxValidator.nonlinearity(sbox),
            'linear_approximation': SBoxValidator.linear_approximation_probability(sbox)
        }
        return results
    
    @staticmethod
    def summarize_validation(validation_results: Dict) -> Tuple[bool, Dict]:
        """
        Summarize validation results.
        
        Args:
            validation_results: Dictionary from full_validation
        
        Returns:
            Tuple of (overall_pass, summary_dict)
        """
        summary = {}
        all_pass = True
        
        for test_name, (passed, message, _) in validation_results.items():
            summary[test_name] = {
                'passed': passed,
                'message': message
            }
            if not passed:
                all_pass = False
        
        return all_pass, summary
