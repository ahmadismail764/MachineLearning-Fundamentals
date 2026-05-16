# Lecture 2: Data Pre-Processing – Part 1

**Dr. Ibrahim Gomaa | Selected Topics 2 in Data Engineering**

---

## Data Classification

| Type                          | Subtypes                                                                                              |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Qualitative (Categorical)** | Nominal (no order, e.g. Gender), Ordinal (ordered, e.g. Blood Group), Binary (2 options, e.g. Yes/No) |
| **Quantitative (Numerical)**  | Discrete (countable, finite), Continuous (measurable on a scale)                                      |

---

## Data Encoding

Converting categorical variables to numerical format. Required because most ML models only accept numbers.

> Data scientists spend **70–80%** of their time on data prep — encoding is a core part.

### Encoding Techniques

#### For Ordinal Data (has inherent order)

| Method               | How it works                                                                        |
| -------------------- | ----------------------------------------------------------------------------------- |
| **Label Encoding**   | Maps each category to an integer preserving rank (e.g. Poor=1, Good=2, Excellent=3) |
| **Ordinal Encoding** | Like Label Encoding but with custom mapping via domain knowledge                    |

#### For Nominal Data (no inherent order)

| Method                 | How it works                                                          | Key Note                                         |
| ---------------------- | --------------------------------------------------------------------- | ------------------------------------------------ |
| **One-Hot Encoding**   | Creates N binary columns; 1 = present, 0 = absent                     | Causes multicollinearity                         |
| **Dummy Encoding**     | Creates N-1 binary columns; one category is the reference (all zeros) | Avoids multicollinearity                         |
| **Frequency Encoding** | Replaces category with its count/proportion in dataset                | Can lose info if two categories share same count |
| **Target Encoding**    | Replaces category with mean of target variable per category           | Risk of overfitting / data leakage               |

#### One-Hot / Dummy Drawbacks

- **Increased dimensionality** (curse of dimensionality with high-cardinality features)
- **No new information** added, just restructured
- **Multicollinearity** — One-Hot creates perfect collinearity; Dummy mitigates but doesn't fully eliminate it

---

## Imputation

Replacing missing values with substitutes instead of deleting rows/columns.

**Why not just delete?** → Data loss, population bias, reduced statistical power.

**Why impute?** → Most ML libraries (e.g. scikit-learn) can't handle NaNs; missing data distorts distributions and biases models.

### Missing Values Treatment Overview

```
Missing Values
├── Deletion
│   ├── Pairwise (delete only missing cells)
│   └── Listwise (delete entire rows)
└── Imputation
    ├── Drop entire columns
    └── Fill values
        ├── General (non-time-series): Constant, Mean/Median/Mode
        └── Advanced
            ├── Time Series: Forward Fill, Back Fill, Linear Interpolation
            └── KNN-Based / MICE
```

### Imputation Techniques

#### Numerical Variables

| Method                 | Best When                                             | Key Limitation                          |
| ---------------------- | ----------------------------------------------------- | --------------------------------------- |
| **Mean**               | Normal (symmetric) distribution; assumes MCAR         | Distorts variance & covariance          |
| **Median**             | Skewed distribution or outliers present; assumes MCAR | Same as mean                            |
| **Arbitrary/Constant** | Data is MNAR; chosen value outside existing range     | Distorts distribution, creates outliers |
| **End of Tail**        | Fill with min/max of distribution                     | —                                       |
| **Mode**               | Most values cluster around one value                  | Over-represents dominant value          |

#### Categorical Variables

| Method                       | How it works                                | Key Limitation                                       |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------------- |
| **Frequent Category (Mode)** | Fill with most common category; assumes MAR | Over-represents mode; distorts distribution at scale |
| **Add "Missing" Category**   | Create a new explicit "Missing" label       | —                                                    |
