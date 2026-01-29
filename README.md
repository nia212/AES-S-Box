# Cryptographic Application - 

A comprehensive Streamlit-based system for AES-256-CBC encryption with custom S-boxes generated via affine transformation over GF(2), following academic cryptographic standards.

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run app.py
```


## 📊 S-box Validation Metrics

All S-boxes are validated for:

- **Bijectivity**: Is it a permutation of 0-255?
- **Balance**: Are output bits equally distributed?
- **SAC (Strict Avalanche Criterion)**: Do single bit flips affect all output bits with ~50% probability?
- **Differential Uniformity**: How resistant is it to differential cryptanalysis?
- **Nonlinearity**: Distance to affine functions (resistance to linear cryptanalysis)
- **Linear Approximation Probability**: Bias in linear approximations


## 📚 References
https://doi.org/10.1007/s11071-024-10414-3


## 📄 License

Educational and research use only.


