# Cryptographic Application - AES-256-CBC with Custom S-boxes

A comprehensive Streamlit-based system for AES-256-CBC encryption with custom S-boxes generated via affine transformation over GF(2), following academic cryptographic standards.

## 🔐 Features

- **AES-256-CBC Encryption** with PBKDF2-HMAC-SHA256 (200,000 iterations)
- **Custom S-box Generation** via bijective affine transformation over GF(2)
- **AES Standard S-box** for comparison and proven security
- **Text & Document Encryption** with full encryption/decryption support
- **Image Encryption** with pixel-level S-box substitution
- **Comprehensive Cryptographic Analysis**:
  - Shannon Entropy (plaintext vs ciphertext)
  - Strict Avalanche Criterion (SAC)
  - Nonlinearity computation
  - Differential Uniformity (DDT & LAT)
  - NPCR/UACI for image cryptography
  - Linear Approximation Probability
  - Balance property validation

## 📂 Project Structure

```
imageCryptoProject/
├── app.py                          # Streamlit UI (NO cryptographic logic)
│
├── core/                           # ALL cryptographic modules
│   ├── __init__.py
│   ├── affine.py                  # Affine transformation over GF(2)
│   ├── text_crypto.py             # Text/document encryption
│   ├── image_crypto.py            # Image encryption with S-box
│   ├── sbox_io.py                 # S-box I/O and presets
│   └── validation.py              # Cryptographic property validation
│
├── metrics/                        # Cryptographic analysis modules
│   ├── __init__.py
│   ├── entropy.py                 # Shannon entropy metrics
│   ├── sac.py                     # Strict Avalanche Criterion
│   ├── nl.py                      # Nonlinearity analysis
│   ├── du.py                      # Differential Uniformity & LAT
│   ├── npcr.py                    # NPCR/UACI for images
│   └── uaci.py                    # Image quality metrics
│
├── sboxes/                         # S-box storage (JSON/Excel)
│
├── tests/                          # Unit tests (NO Streamlit)
│   ├── __init__.py
│   └── test_crypto.py             # Comprehensive test suite
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### 3. Run Tests

```bash
pytest tests/test_crypto.py -v
```

## 📊 Key Modules

### Core Cryptographic Logic

#### `affine.py` - Affine Transformation
- GF(2^8) arithmetic (multiplication, inversion)
- Matrix operations over GF(2)
- Bijective affine S-box generation: `y = Ax + b`
- Invertibility verification
- AES standard S-box (reference)

```python
from core.affine import generate_affine_sbox, verify_matrix_invertibility_gf2
import numpy as np

# Create transformation matrix and constant
matrix = np.eye(8, dtype=int)
constant = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=int)

# Verify invertibility
if verify_matrix_invertibility_gf2(matrix):
    sbox = generate_affine_sbox(matrix, constant)
```

#### `text_crypto.py` - Text Encryption
- AES-256-CBC encryption/decryption
- PBKDF2-HMAC-SHA256 key derivation (200,000 iterations)
- PKCS7 padding
- Document file support (.txt, .md, .csv, .json, .xml, .log)

```python
from core.text_crypto import TextEncryptor

# Encrypt
encrypted = TextEncryptor.encrypt("Secret message", "password")

# Decrypt
plaintext = TextEncryptor.decrypt(encrypted, "password")
```

#### `image_crypto.py` - Image Encryption
- S-box substitution on pixel values
- Support for RGB and grayscale images
- Image comparison metrics
- Histogram analysis

```python
from core.image_crypto import ImageEncryptor
import numpy as np

# Encrypt image
encrypted_data = ImageEncryptor.encrypt_image("image.png", sbox)

# Decrypt image
decrypted_array, mode = ImageEncryptor.decrypt_image(encrypted_data, inv_sbox)
```

#### `sbox_io.py` - S-box Management
- JSON/Excel save and load
- Matrix presets: Identity, AES, K44, K43, K45
- Batch operations
- Comparison reports

```python
from core.sbox_io import SBoxIO

# Get preset matrix
matrix = SBoxIO.get_matrix_preset('aes')

# Save S-box
filepath = SBoxIO.save_sbox_excel(sbox, "my_sbox", metrics_dict)

# Load S-box
loaded_sbox = SBoxIO.load_sbox_json("my_sbox.json")
```

#### `validation.py` - Cryptographic Validation
- Bijectivity check
- Balance property verification
- SAC (Strict Avalanche Criterion)
- Differential Uniformity
- Nonlinearity
- Linear Approximation Probability

```python
from core.validation import SBoxValidator

results = SBoxValidator.full_validation(sbox)
all_pass, summary = SBoxValidator.summarize_validation(results)
```

### Metrics Modules

#### `entropy.py`
```python
from metrics.entropy import shannon_entropy, entropy_analysis

entropy_val = shannon_entropy(data)
analysis = entropy_analysis(plaintext, ciphertext)
```

#### `sac.py` - Strict Avalanche Criterion
```python
from metrics.sac import compute_sac_matrix

sac_matrix = compute_sac_matrix(sbox)  # 8x8 probability matrix
```

#### `nl.py` - Nonlinearity
```python
from metrics.nl import sbox_nonlinearity

nl_info = sbox_nonlinearity(sbox)
# Returns min, max, avg, std per-bit nonlinearity
```

#### `du.py` - Differential Uniformity
```python
from metrics.du import differential_uniformity, linear_approximation_probability

du_info = differential_uniformity(sbox)
lap_info = linear_approximation_probability(sbox)
```

#### `npcr.py` & `uaci.py` - Image Analysis
```python
from metrics.npcr import npcr_uaci_analysis
from metrics.uaci import image_quality_metrics, histogram_metrics

metrics = npcr_uaci_analysis(plaintext_img, ciphertext_img)
quality = image_quality_metrics(original, encrypted)
hist = histogram_metrics(original, encrypted)
```

## 🧪 Testing

Comprehensive unit tests without Streamlit dependencies:

```bash
pytest tests/test_crypto.py -v
```

Test Coverage:
- Affine transformation (GF(2^8))
- Text encryption roundtrips
- S-box I/O operations
- Validation functions
- Metrics computation
- Integration tests

## 📋 Cryptographic Standards

| Property | Value |
|----------|-------|
| Encryption Algorithm | AES-256-CBC |
| Key Derivation | PBKDF2-HMAC-SHA256 |
| PBKDF2 Iterations | 200,000 |
| Key Size | 256 bits (32 bytes) |
| IV Size | 128 bits (16 bytes) |
| Block Size | 128 bits (16 bytes) |
| Padding | PKCS7 |
| S-box Type | Custom Affine or Standard AES |

## 🔒 Security Notes

1. **Custom S-boxes** are treated as AES-like, not standard AES
2. For production systems, use **AES standard S-box** which has proven security
3. This application is for **educational and research purposes**
4. Always test thoroughly before production use
5. Use strong, unique passwords
6. Keep IV/salt private and never reuse

## 📊 S-box Validation Metrics

All S-boxes are validated for:

- **Bijectivity**: Is it a permutation of 0-255?
- **Balance**: Are output bits equally distributed?
- **SAC (Strict Avalanche Criterion)**: Do single bit flips affect all output bits with ~50% probability?
- **Differential Uniformity**: How resistant is it to differential cryptanalysis?
- **Nonlinearity**: Distance to affine functions (resistance to linear cryptanalysis)
- **Linear Approximation Probability**: Bias in linear approximations

## 🎨 UI Features

### S-box Generation
- Custom affine transformation
- Matrix presets (Identity, AES, K44, K43, K45)
- Real-time bijectivity checking
- Visual matrix input

### S-box Validation
- Full validation suite
- Per-test detailed results
- Comparison with AES S-box
- Summary statistics

### Text Encryption
- Encrypt/decrypt text messages
- Password-based key derivation
- JSON export format
- Support for documents

### Image Encryption
- Upload and encrypt images
- Side-by-side comparison
- NPCR/UACI metrics
- Histogram analysis
- Quality metrics (MSE, PSNR, Correlation)

### Batch Operations
- Export S-boxes to Excel
- Create comparison reports
- Save metrics alongside S-boxes

## 📈 Example Usage

### Generate Custom S-box
```python
from core.affine import generate_affine_sbox
from core.validation import SBoxValidator
import numpy as np

# Create affine transformation
matrix = np.eye(8, dtype=int)
constant = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=int)

# Generate S-box
sbox = generate_affine_sbox(matrix, constant)

# Validate
results = SBoxValidator.full_validation(sbox)
all_pass, summary = SBoxValidator.summarize_validation(results)
```

### Encrypt Text
```python
from core.text_crypto import TextEncryptor

plaintext = "Secret message"
password = "strong_password"

# Encrypt
encrypted = TextEncryptor.encrypt(plaintext, password)
print(f"Ciphertext: {encrypted['ciphertext']}")
print(f"IV: {encrypted['iv']}")
print(f"Salt: {encrypted['salt']}")

# Decrypt
decrypted = TextEncryptor.decrypt(encrypted, password)
assert decrypted == plaintext
```

### Encrypt Image
```python
from core.image_crypto import ImageEncryptor
from metrics.npcr import npcr_uaci_analysis

# Encrypt
encrypted_data = ImageEncryptor.encrypt_image("photo.jpg", sbox)

# Decrypt
inv_sbox = ImageEncryptor.compute_inverse_sbox(sbox)
decrypted_array, mode = ImageEncryptor.decrypt_image(encrypted_data, inv_sbox)

# Analyze
metrics = npcr_uaci_analysis(plaintext_array, encrypted_data['encrypted_image'])
print(f"NPCR: {metrics['npcr']:.2f}%")
print(f"UACI: {metrics['uaci']:.2f}%")
```

## 🛠️ Development

### Code Quality
- Clean code principles
- Pure functions for all cryptographic operations
- No global state in core modules
- Strong input validation
- Deterministic and reproducible results

### Modular Architecture
- Separation of concerns: UI (app.py) vs Logic (core/ + metrics/)
- Reusable, testable modules
- No cryptographic logic in Streamlit code
- Independent metric computation

## 📚 References

- FIPS 197: Advanced Encryption Standard (AES)
- FIPS 198-1: The Keyed-Hash Message Authentication Code (HMAC)
- NIST SP 800-132: Password-Based Key Derivation Function (PBKDF)
- RFC 3394: Advanced Encryption Standard (AES) Key Wrap Algorithm

## 📄 License

Educational and research use only.

## 👨‍💻 Author

Built as a comprehensive cryptographic system with academic rigor and clean architecture.

---

**Last Updated**: January 2026
**Version**: 1.0
