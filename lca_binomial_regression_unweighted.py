"""
Binary Logistic Regression on LCA Class Membership (Class 2 vs. Class 1)
========================================================================

This script performs a binary logistic regression to predict Class 2 (Globally Low Engagers)
versus Class 1 (Broadly Moderate Preventers) based on predictors that were found to be 
significantly associated in prior bivariate analyses (Table 2).

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# Load the dataset
df = pd.read_excel("data_ghp16_lca_macrodata_analysis_subset.xlsx")

# Filter for Class 1 and 2 only and create binary outcome
df = df[df['LCA_class'].isin([1, 2])].copy()
df['binary_class'] = df['LCA_class'].replace({1: 0, 2: 1})  # Class 2 = 1 (event), Class 1 = 0

# Define variables based on prior significant bivariate associations
selected_vars = [
    'GenderM0F1', 'AgeGroup', 'Education_cat', 'has_chronic_disease',
    'Mistrust_Index', 'Lives_Alone_binary', 'PAM_level_cat'
]

# Convert appropriate columns to categorical
categorical_vars = ['GenderM0F1', 'AgeGroup', 'Education_cat', 'Lives_Alone_binary', 'PAM_level_cat']
for var in categorical_vars:
    df[var] = df[var].astype('category')

# Define formula for logistic regression
formula = 'binary_class ~ C(GenderM0F1) + C(AgeGroup) + C(Education_cat) + has_chronic_disease + Mistrust_Index + C(Lives_Alone_binary) + C(PAM_level_cat)'

# Fit the model
model = smf.logit(formula=formula, data=df)
result = model.fit()

# Print model summary
print(result.summary2())

# Compute VIFs
y, X = dmatrices(formula, df, return_type='dataframe')
vif_data = pd.DataFrame({
    'Variable': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("\nVariance Inflation Factors:")
print(vif_data)

# Output model statistics
print(f"\nNagelkerke R² approximation: {result.prsquared:.3f}")
print(f"Likelihood ratio test: chi2 = {result.llr:.2f}, p = {result.llr_pvalue:.4f}")
