
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import os


class ImageEncryptor:
    """Handle image encryption/decryption with S-box substitution."""
    
    @staticmethod
    def load_image(image_path: str) -> Tuple[np.ndarray, str]:
        """
        Load an image and return as numpy array.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (image array, color mode)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            img = Image.open(image_path)
            mode = img.mode
            img_array = np.array(img)
            return img_array, mode
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    @staticmethod
    def save_image(image_array: np.ndarray, output_path: str, mode: str = 'RGB') -> None:
        """
        Save image array to file.
        
        Args:
            image_array: Image as numpy array
            output_path: Output file path
            mode: PIL image mode (RGB, L, etc.)
        """
        # Ensure array is proper format
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(image_array, mode=mode)
        img.save(output_path)
    
    @staticmethod
    def encrypt_image(image_path: str, sbox: np.ndarray) -> Dict[str, Any]:
        """
        Encrypt image using S-box substitution + pixel scrambling for strong diffusion.
        
        Args:
            image_path: Path to input image
            sbox: S-box for substitution (256 values)
        
        Returns:
            Dictionary with encrypted image data and metadata
        """
        if len(sbox) != 256:
            raise ValueError("S-box must have 256 entries")
        
        # Load image
        image_array, mode = ImageEncryptor.load_image(image_path)
        original_shape = image_array.shape
        
        # Step 1: Flatten the image
        flat_image = image_array.flatten().astype(np.uint8)
        
        # Step 2: Apply S-box substitution (confusion)
        substituted = np.array([sbox[pixel] for pixel in flat_image], dtype=np.uint8)
        
        # Step 3: Apply pixel scrambling/permutation (diffusion)
        # Use a pseudo-random permutation based on S-box values
        n_pixels = len(substituted)
        permutation = np.arange(n_pixels)
        
        # Generate permutation from S-box values
        seed_values = substituted[:min(256, n_pixels)]
        for i in range(n_pixels):
            j = (i + seed_values[i % len(seed_values)]) % n_pixels
            permutation[i], permutation[j] = permutation[j], permutation[i]
        
        # Apply permutation
        scrambled = substituted[permutation]
        
        # Step 4: Apply diffusion by XORing with neighbors
        diffused = scrambled.copy()
        for i in range(1, len(diffused)):
            diffused[i] = (diffused[i] ^ diffused[i-1]) % 256
        
        # Step 5: Reshape back to original shape
        encrypted_array = diffused.reshape(original_shape)
        
        return {
            'encrypted_image': encrypted_array,
            'original_shape': original_shape,
            'mode': mode,
            'filename': os.path.basename(image_path),
            'sbox': sbox.tolist(),
            'permutation': permutation.tolist()
        }
    
    @staticmethod
    def decrypt_image(encrypted_data: Dict[str, Any], inverse_sbox: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Decrypt image by reversing S-box substitution, diffusion, and permutation.
        
        Args:
            encrypted_data: Dictionary with encrypted image data
            inverse_sbox: Inverse S-box for decryption (256 values)
        
        Returns:
            Tuple of (decrypted image array, mode)
        """
        if len(inverse_sbox) != 256:
            raise ValueError("Inverse S-box must have 256 entries")
        
        encrypted_array = encrypted_data['encrypted_image']
        mode = encrypted_data['mode']
        original_shape = encrypted_data['original_shape']
        permutation = np.array(encrypted_data.get('permutation', []))
        
        # Flatten the encrypted image
        flat_encrypted = encrypted_array.flatten().astype(np.uint8)
        
        # Step 1: Reverse diffusion (undo XOR with neighbors)
        # If encryption does: enc[i] = (val[i] ^ enc[i-1])
        # Then decryption does: val[i] = enc[i] ^ enc[i-1]
        undiffused = np.zeros_like(flat_encrypted)
        undiffused[0] = flat_encrypted[0]
        for i in range(1, len(undiffused)):
            undiffused[i] = (flat_encrypted[i] ^ flat_encrypted[i-1]) % 256
        
        # Step 2: Reverse permutation (if available)
        if len(permutation) > 0:
            unscrambled = np.zeros_like(undiffused)
            for i in range(len(permutation)):
                unscrambled[permutation[i]] = undiffused[i]
        else:
            unscrambled = undiffused
        
        # Step 3: Apply inverse S-box (reverse substitution)
        decrypted = np.array([inverse_sbox[pixel] for pixel in unscrambled], dtype=np.uint8)
        
        # Step 4: Reshape back to original shape
        decrypted_array = decrypted.reshape(original_shape)
        
        return decrypted_array, mode
    
    @staticmethod
    def _apply_sbox(image_array: np.ndarray, sbox: np.ndarray) -> np.ndarray:
        """
        Apply S-box substitution to all pixel values.
        
        Args:
            image_array: Image array
            sbox: S-box for substitution
        
        Returns:
            Transformed image array
        """
        # Flatten, apply S-box, reshape
        original_shape = image_array.shape
        flat = image_array.flatten().astype(np.uint16)  # uint16 to avoid overflow during indexing
        
        # Ensure indices are in valid range
        flat = np.minimum(flat, 255)
        transformed = sbox[flat].astype(np.uint8)
        
        return transformed.reshape(original_shape)
    
    @staticmethod
    def compute_inverse_sbox(sbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute inverse S-box from forward S-box.
        
        Args:
            sbox: Forward S-box
        
        Returns:
            Inverse S-box or None if not bijective
        """
        if len(sbox) != 256:
            raise ValueError("S-box must have 256 entries")
        
        # Check bijectivity
        if len(np.unique(sbox)) != 256:
            return None  # Not bijective
        
        # Create inverse mapping
        inv_sbox = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            inv_sbox[sbox[i]] = i
        
        return inv_sbox
    
    @staticmethod
    def compare_images(image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """
        Compare two images and return similarity metrics.
        
        Args:
            image1: First image array
            image2: Second image array
        
        Returns:
            Dictionary with NPCR and UACI metrics
        """
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same shape")
        
        # Number of Pixels Change Rate (NPCR)
        pixels_changed = np.sum(image1 != image2)
        total_pixels = image1.size
        npcr = (pixels_changed / total_pixels) * 100
        
        # Unified Average Changing Intensity (UACI)
        diff = np.abs(image1.astype(float) - image2.astype(float))
        uaci = (np.sum(diff) / (total_pixels * 255)) * 100
        
        return {
            'npcr': npcr,
            'uaci': uaci,
            'pixels_changed': int(pixels_changed),
            'total_pixels': total_pixels
        }
    
    @staticmethod
    def get_histogram(image_array: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram of image pixel values.
        
        Args:
            image_array: Image array
            bins: Number of histogram bins
        
        Returns:
            Tuple of (histogram, bin_edges)
        """
        flat = image_array.flatten().astype(np.uint8)
        histogram, bin_edges = np.histogram(flat, bins=bins, range=(0, 256))
        return histogram, bin_edges
    
    @staticmethod
    def extract_channel(image_array: np.ndarray, channel: int) -> np.ndarray:
        """
        Extract a single channel from RGB image.
        
        Args:
            image_array: Image array (must be RGB)
            channel: Channel index (0=R, 1=G, 2=B)
        
        Returns:
            Single channel array
        """
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Image must be RGB (3 channels)")
        
        if channel not in [0, 1, 2]:
            raise ValueError("Channel must be 0 (R), 1 (G), or 2 (B)")
        
        return image_array[:, :, channel]
