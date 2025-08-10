# Breast Cancer Wisconsin Classification

This project implements a **machine learning pipeline** to classify tumors as **Benign** or **Malignant** using the **Breast Cancer Wisconsin dataset**.  
It covers **data cleaning**, **exploratory analysis**, **class balancing (SMOTE)**, and evaluation of models in both **Scikit-learn** and **PyTorch**, with a focus on maximizing recall for medical diagnosis.

---

## Project Overview
- **Goal:** Predict tumor type (`0` = Benign, `1` = Malignant) from cytological features.
- **Dataset:** Breast Cancer Wisconsin dataset — 449 samples, 9 integer-valued features.
- **Challenge:** Class imbalance (~63% benign, ~37% malignant) and minimizing false negatives.
- **Approach:** Data preprocessing, SMOTE balancing, model comparison, PyTorch implementation with regularization.

---

## Steps & Methodology

### 1. Data Preprocessing
- **Missing Values:** Imputed using `IterativeImputer` to preserve multivariate relationships.
- **Class Encoding:** `2 → 0` (Benign), `4 → 1` (Malignant).
- **Outlier Removal:** IQR-based filtering reduced dataset from 449 → 317 samples.
- **Feature Scaling:** All features scaled to `[0,1]` using `MinMaxScaler`.
- **Train-Test Split:** 80% train / 20% test with stratification.

### 2. Exploratory Data Analysis (EDA)
- **Feature Distributions:** Malignant tumors generally have higher values in `Clump_thickness`, `Uniformity_of_cell_size`, etc.
- **Correlation:** Strong correlations observed (`Uniformity_of_cell_size` ↔ `Uniformity_of_cell_shape`: r = 0.88).
- **Class Imbalance:** Resolved using **SMOTE** on training data.

### 3. Models Trained
- **Scikit-learn Models:**
  - Perceptron (baseline)
  - MLP (default & tuned via GridSearchCV)
- **PyTorch Models:**
  - Custom MLP architecture
  - Tuned hyperparameters (hidden layers, activation, learning rate)
  - Regularization experiments (Dropout, L2 weight decay)

---

## Final Results

### Performance Summary (Test Set)
While all models achieved **AUC > 0.97**, the most important factor in a medical setting is **recall for the malignant class**.

- **Best Scikit-learn Model:** Tuned MLP — 83.3% recall, AUC 0.9771.
- **Best PyTorch Model (No Reg):** Recall 96%, strong precision and F1.
- **PyTorch + Regularization:**  
  - **Dropout (p=0.5)** → **100% recall** (zero false negatives), Accuracy 95.3%, F1-score 0.9412, AUC 0.9844.
  - L2 regularization also boosted performance, but Dropout achieved the best trade-off.

**Champion Model:**  
> **PyTorch MLP** — single hidden layer (8 neurons), LeakyReLU activation, learning rate 0.001, Dropout rate 0.5.

**Why It Wins:**
- **Perfect Recall (100%)** — no malignant tumors missed.
- **Highest Accuracy** among perfect-recall models (95.3%).
- **Best F1-Score** (0.9412) — ideal precision/recall balance.
- **Top AUC** (0.9844) — excellent class separation ability.
