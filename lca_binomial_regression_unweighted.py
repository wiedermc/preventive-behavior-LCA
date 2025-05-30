"""
Binary Logistic Regression on LCA Class Membership (Unweighted)
================================================================

This script performs an unweighted binary logistic regression analysis 
on LCA-derived class membership to identify predictors of low preventive 
health engagement. Specifically, Class 2 (Globally Low Engagers) is 
compared to Class 1 (Broadly Moderate Preventers).

Inputs:
- A dataset containing 'LCA_class' and all predictor variables.
- Categorical predictors should follow the names used in the script.

Outputs:
- Console output of model summary
- VIF table for multicollinearity check (optional export)

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# Load data
df = pd.read_excel("gesundheitskompetenz_with_LCA_classes.xlsx")

# Filter for binary comparison: Class 2 (code = 2) vs. Class 1 (code = 1)
df = df[df['LCA_class'].isin([1, 2])].copy()
df['binary_class'] = df['LCA_class'].replace({1: 0, 2: 1})  # Class 2 = 1 (event), Class 1 = 0

# Set correct dtypes for categorical variables
categoricals = ['GenderM0F1', 'AgeGroup', 'Education_cat', 'Language', 
                'HLS_score_cat', 'PAM_level_cat', 'Lives_Alone_binary', 
                'Workinthehealthorsocialsector']
for var in categoricals:
    df[var] = df[var].astype('category')

# Define formula
formula = 'binary_class ~ C(GenderM0F1) + C(AgeGroup) + C(Education_cat) + C(Language) + \
has_chronic_disease + C(HLS_score_cat) + C(PAM_level_cat) + Mistrust_Index + \
C(Lives_Alone_binary) + C(Workinthehealthorsocialsector)'

# Fit logistic regression model
model = smf.logit(formula=formula, data=df)
result = model.fit()
print(result.summary2())

# VIF calculation
y, X = dmatrices(formula, data=df, return_type='dataframe')
vif_data = pd.DataFrame({
    'Variable': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("\nVariance Inflation Factors (VIF):")
print(vif_data)
