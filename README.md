# Cryptographic Application - 

A comprehensive Streamlit-based system for AES-256-CBC encryption with custom S-boxes generated via affine transformation over GF(2), following academic cryptographic standards.

## 🚀 Quick Start

### 1. Clone repository
```bash
git clone https://github.com/nia212/AES-S-Box.git
```

### 2. Installation

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```


## 📊 S-box Validation Metrics

All S-boxes are validated for:

- **Nonlinearity**
- **Strict Avalanche Criterion (SAC)**
- **Bit Independence Criterion (BIC)** 
- **Linear Approximation Probability (LAP)** 
- **Differential Approximation Probability (DAP)** m
- **Entropy analysis** 
- **Number of Pixel Change Rate (NPCR)** 
- **Unified Average Changing Intensity (UACI)** 
- **Correlation Immunity (CI)** 
- **Algebraic Degree (AD)** 
- **Histogram Analysis**
  


## 📚 References
https://doi.org/10.1007/s11071-024-10414-3



## 📄 License

Educational and research use only.


