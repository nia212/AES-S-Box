
import os
import json
import base64
from typing import Tuple, Dict, Any
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
import numpy as np


class TextEncryptor:
    """Handle text encryption/decryption with AES-256-CBC."""
    
    # AES-256 requires 32-byte keys
    KEY_SIZE = 32
    # CBC mode requires 16-byte IVs
    IV_SIZE = 16
    # PBKDF2 iterations as per security standards
    PBKDF2_ITERATIONS = 200000
    
    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2-HMAC-SHA256.
        
        Args:
            password: User password
            salt: Random salt (should be 16+ bytes)
        
        Returns:
            32-byte encryption key
        """
        if not isinstance(password, str):
            raise ValueError("Password must be a string")
        
        if len(salt) < 8:
            raise ValueError("Salt must be at least 8 bytes")
        
        key = PBKDF2(
            password,
            salt,
            dkLen=TextEncryptor.KEY_SIZE,
            count=TextEncryptor.PBKDF2_ITERATIONS,
            hmac_hash_module=SHA256
        )
        return key
    
    @staticmethod
    def encrypt(plaintext: str, password: str, iv: bytes = None, sbox: np.ndarray = None) -> Dict[str, str]:
        """
        Encrypt plaintext using AES-256-CBC.
        
        Args:
            plaintext: Text to encrypt
            password: Encryption password
            iv: Optional 16-byte IV (auto-generated if None)
            sbox: Optional custom S-box (used as metadata, not in standard AES)
        
        Returns:
            Dictionary with encrypted data in hex format:
            {
                'ciphertext': hex string,
                'iv': hex string,
                'salt': hex string,
                'sbox_name': name of S-box if provided
            }
        """
        if not plaintext:
            raise ValueError("Plaintext cannot be empty")
        
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Generate random salt and IV
        salt = os.urandom(16)
        if iv is None:
            iv = os.urandom(TextEncryptor.IV_SIZE)
        
        if len(iv) != TextEncryptor.IV_SIZE:
            raise ValueError(f"IV must be {TextEncryptor.IV_SIZE} bytes")
        
        # Derive encryption key
        key = TextEncryptor.derive_key(password, salt)
        
        # Create cipher
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Pad plaintext to AES block size (16 bytes) using PKCS7
        padded_plaintext = TextEncryptor._pad_pkcs7(plaintext.encode('utf-8'))
        
        # Encrypt
        ciphertext = cipher.encrypt(padded_plaintext)
        
        result = {
            'ciphertext': ciphertext.hex(),
            'iv': iv.hex(),
            'salt': salt.hex()
        }
        
        if sbox is not None:
            result['sbox_used'] = 'custom'
        
        return result
    
    @staticmethod
    def decrypt(encrypted_data: Dict[str, str], password: str, sbox: np.ndarray = None) -> str:
        """
        Decrypt ciphertext using AES-256-CBC.
        
        Args:
            encrypted_data: Dictionary with 'ciphertext', 'iv', 'salt' in hex format
            password: Decryption password
            sbox: Optional custom S-box (metadata only)
        
        Returns:
            Decrypted plaintext
        """
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Extract components
        try:
            ciphertext = bytes.fromhex(encrypted_data['ciphertext'])
            iv = bytes.fromhex(encrypted_data['iv'])
            salt = bytes.fromhex(encrypted_data['salt'])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid encrypted data format: {e}")
        
        # Derive key
        key = TextEncryptor.derive_key(password, salt)
        
        # Create cipher
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Decrypt
        padded_plaintext = cipher.decrypt(ciphertext)
        
        # Unpad
        plaintext = TextEncryptor._unpad_pkcs7(padded_plaintext)
        
        return plaintext.decode('utf-8')
    
    @staticmethod
    def _pad_pkcs7(data: bytes) -> bytes:
        """
        Apply PKCS7 padding to data.
        
        Args:
            data: Input data
        
        Returns:
            Padded data
        """
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    @staticmethod
    def _unpad_pkcs7(data: bytes) -> bytes:
        """
        Remove PKCS7 padding from data.
        
        Args:
            data: Padded data
        
        Returns:
            Unpadded data
        """
        padding_length = data[-1]
        
        if padding_length < 1 or padding_length > 16:
            raise ValueError("Invalid PKCS7 padding")
        
        # Verify all padding bytes are correct
        for i in range(padding_length):
            if data[-(i + 1)] != padding_length:
                raise ValueError("Invalid PKCS7 padding")
        
        return data[:-padding_length]


class DocumentEncryptor:
    """Handle document (text files) encryption/decryption."""
    
    SUPPORTED_FORMATS = {'.txt', '.md', '.csv', '.json', '.xml', '.log'}
    
    @staticmethod
    def encrypt_file(file_path: str, password: str, sbox: np.ndarray = None) -> Dict[str, Any]:
        """
        Encrypt a document file.
        
        Args:
            file_path: Path to the document
            password: Encryption password
            sbox: Optional custom S-box
        
        Returns:
            Encrypted file data
        """
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")
        
        # Get file info
        import os.path
        filename = os.path.basename(file_path)
        
        # Encrypt content
        encrypted = TextEncryptor.encrypt(content, password, sbox=sbox)
        encrypted['filename'] = filename
        
        return encrypted
    
    @staticmethod
    def decrypt_file(encrypted_data: Dict[str, Any], password: str, sbox: np.ndarray = None) -> Tuple[str, str]:
        """
        Decrypt a document file.
        
        Args:
            encrypted_data: Encrypted file data
            password: Decryption password
            sbox: Optional custom S-box
        
        Returns:
            Tuple of (filename, content)
        """
        filename = encrypted_data.get('filename', 'decrypted.txt')
        content = TextEncryptor.decrypt(encrypted_data, password, sbox=sbox)
        return filename, content
