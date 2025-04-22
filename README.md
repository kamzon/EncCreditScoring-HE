# EncCreditScoring-HE

EncCreditScoring-HE is a cutting-edge, privacy-preserving credit scoring system built using Homomorphic Encryption (HE).  
This project leverages the SEAL library (developed by Microsoft Research) and the CKKS encryption scheme to perform secure computations on encrypted financial data, ensuring that sensitive user information remains private throughout the entire scoring and inference process.

---

## ✨ Features

- Privacy-preserving multi-class credit scoring
- Fully encrypted inference using CKKS homomorphic encryption
- Comparison between encrypted inference and plaintext inference
- Polynomial approximation for efficient encrypted sigmoid evaluation
- Modular and professional C++ project structure
- Environment-driven configuration via `.env` file
- Based on Microsoft SEAL v4.1.0
- Ready for research extension and academic publication

---


## ⚙️ Requirements

- **C++17** or higher
- **Microsoft SEAL** library (v4.1.0)
- **Microsoft GSL** (Guideline Support Library v3.1.0)
- **CMake** (>= 3.10)

Make sure Microsoft SEAL is properly compiled and available.  
Microsoft GSL must be downloaded separately as a header-only library.

---

## 📦 Installation

### 1. Clone the repository

```bash
git git@github.com:kamzon/EncCreditScoring-HE.git
cd EncCreditScoring-HE/cpp
```

## 📂 Setting Up .env File

Create a .env file inside the cpp/ directory with the following content:

```bash
WEIGHT_MATRIX_PATH=/absolute/path/to/weight_matrix.csv
BIAS_VECTOR_PATH=/absolute/path/to/bias_vector.csv
X_TEST_PATH=/absolute/path/to/X_test.csv
Y_TEST_PATH=/absolute/path/to/y_test.csv
```

## 🛠 Build Instructions

From the cpp/ directory:

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./HomomorphicLogisticRegression
```
