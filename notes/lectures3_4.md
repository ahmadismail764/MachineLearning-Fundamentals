# Lecture 3: Data Pre-Processing – Part 2
**Scaling & Resampling**

---

## Feature Scaling
Normalizing features into a consistent range so no single feature dominates model training due to its larger numerical scale.

### Scaling Techniques

| Technique | Method | Formula | Output Range | Best When | Outlier Sensitivity |
|---|---|---|---|---|---|
| **Standard Scaler** | Subtract mean, divide by std | `(X - μ) / σ` | Unbounded (~-3 to +3) | Data is Gaussian | Low |
| **Min-Max Scaler** | Subtract min, divide by range | `(X - Xmin) / (Xmax - Xmin)` | 0 to 1 | Features have different scales | **High** |
| **Mean Normalization** | Subtract mean, divide by range | `(X - μ) / (Xmax - Xmin)` | -1 to 1 | Similar to Min-Max but centered | High |
| **Absolute Max Scaler** | Divide by absolute maximum | `X / max(|X|)` | -1 to 1 | Sparse data, +/- values | High |
| **Robust Scaler** | Subtract median, divide by IQR | `(X - Median) / IQR` | Unbounded | Data has **outliers** | **Low** |

---

## Data Resampling
Addresses **imbalanced datasets** (e.g. 95% majority vs 5% minority class) where models become biased toward the majority class.

### Resampling Techniques

| Technique | How it works | ✅ Advantages | ❌ Disadvantages |
|---|---|---|---|
| **Random Under-Sampling** | Remove random majority class samples | Faster training, less storage | Discards useful data; may introduce bias |
| **Random Over-Sampling** | Duplicate minority class samples | No information loss | Risk of overfitting; longer training |
| **SMOTE** | Generate *synthetic* minority samples using feature space interpolation | Reduces overfitting vs. random oversampling; no info loss | Ignores class boundaries; may cause class overlap or introduce noise |

---
---

# Lecture 4: Data Pre-Processing – Part 3
**Feature Selection & Feature Reduction**

---

## Overview: Feature Reduction Techniques
```
Feature Reduction Techniques
├── Feature Selection (keeps original features, selects a subset)
│   ├── Filter Methods       → Information Gain, Correlation Coefficient, Variance Threshold
│   ├── Wrapper Methods      → Forward Selection, Backward Elimination, RFE
│   └── Embedded Methods     → LASSO (L1), Random Forest Importance
└── Dimension Reduction (transforms into new lower-dim space)
    ├── PCA (unsupervised)
    └── LDA (supervised)
```

---

## Feature Selection
Finding the best subset of features to build optimized models. Can be **supervised** (labeled data) or **unsupervised** (unlabeled data).

### Filter Methods
Rank features using univariate statistics — independent of any model. Fast and cheap.

| Method | How it works | Key Limitation |
|---|---|---|
| **Information Gain** | Measures reduction in entropy per feature relative to target | — |
| **Correlation Coefficient** | Keeps features highly correlated with target but uncorrelated with each other (Pearson) | Only captures linear relationships |
| **Variance Threshold** | Drops features with near-zero variance (nearly constant values) | Ignores relationship between features and target |

### Wrapper Methods
Train a model repeatedly on different feature subsets — higher accuracy but computationally expensive.

| Method | Direction | How it works |
|---|---|---|
| **Forward Selection** | Bottom-up | Start empty → add best feature one at a time |
| **Backward Elimination** | Top-down | Start with all → remove least significant one at a time |
| **RFE** (Recursive Feature Elimination) | Iterative | Rank by importance weights → repeatedly prune weakest features |

### Embedded Methods
Feature selection built into model training — balances accuracy and cost.

| Method | How it works |
|---|---|
| **LASSO (L1 Regularization)** | Adds L1 penalty that shrinks unimportant feature coefficients to exactly **zero**, effectively removing them |
| **Random Forest Importance** | Ranks features by how much they reduce Gini impurity across all trees in the ensemble |

---

## Feature Reduction (Dimensionality Reduction)
Transforms p dimensions into k dimensions (k << p) while preserving as much information as possible. Reduces storage, speeds training, handles redundancy, and enables visualization.

| Technique | Type | How it works |
|---|---|---|
| **PCA** (Principal Component Analysis) | Unsupervised | Projects data onto new axes (Principal Components) that capture maximum variance. Components are linear combinations of original features — orthogonal and uncorrelated. |
| **LDA** (Linear Discriminant Analysis) | Supervised | Maximizes class separation by finding axes that best distinguish group boundaries. Best for classification tasks. |

### PCA Steps
1. **Standardize** all features
2. Compute **covariance/correlation matrix**
3. **Eigen-decomposition** → get eigenvectors (directions) and eigenvalues (variance explained)
4. **Sort** components by eigenvalue descending → select top k components