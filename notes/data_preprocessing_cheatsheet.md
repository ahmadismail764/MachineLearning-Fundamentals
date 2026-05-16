# Data Preprocessing Cheat Sheet: Lab 2 Pipeline

A simplified, sequential reference for the core data preprocessing workflow explored in Lab 2.

### Step 1: Initial Data Inspection
**Goal:** Understand the dimensions and structure of the dataset.
```python
# View dimensions (rows, columns)
loan_data.shape

# List all feature names
loan_data.columns

# Summary of data types and non-null counts
loan_data.info()
```

### Step 2: Quantitative Exploration
**Goal:** Identify central tendencies, dispersion, and potential outliers in numerical features.
```python
# Generate descriptive statistics for numerical columns
loan_data.describe()
```

### Step 3: Removing Irrelevant Features
**Goal:** Drop columns that do not contribute predictive value to the analysis.
```python
# Remove specific columns (e.g., ID and constant values)
loan_data.drop(['ID', 'year'], axis=1, inplace=True)
```

### Step 4: Qualitative Exploration
**Goal:** Analyze the distribution and frequency of categories within object-type columns.
```python
# Statistics for categorical features
loan_data.describe(include='object')

# Frequency count of a specific category
loan_data['age'].value_counts()

# Proportional distribution (percentage)
loan_data['Gender'].value_counts(normalize=True)
```

### Step 5: Handling Duplicate Data
**Goal:** Ensure data integrity by identifying and removing exact duplicate rows.
```python
# Count total duplicate rows
loan_data.duplicated().sum()

# Remove duplicates from the DataFrame
loan_data.drop_duplicates(inplace=True)

# Reset index after removal
loan_data.reset_index(inplace=True)
```

### Step 6: Identifying Missing Values
**Goal:** Detect null entries to determine if imputation or row deletion is required.
```python
# Count missing values per feature
loan_data.isna().sum()
```

### Step 7: Visual Feature Analysis
**Goal:** Visualize distributions and relationships between pairs of features.
```python
# Matrix of scatter plots for numerical features
sns.pairplot(loan_data)

# Count plot for a single categorical feature
sns.countplot(data=loan_data, x='loan_type')
```

### Step 8: Correlation Analysis
**Goal:** Identify linear relationships between numerical features to detect multicollinearity.
```python
# Compute correlation matrix and visualize as a heatmap
corr = loan_data.corr()
sns.heatmap(corr, annot=True)
```
