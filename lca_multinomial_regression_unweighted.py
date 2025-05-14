
# lca_multinomial_regression_unweighted.py

"""
Multinomial Logistic Regression on LCA Class Membership (Unweighted)
====================================================================

This script performs an unweighted multinomial logistic regression analysis using LCA-derived classes
of preventive health behavior as the outcome variable. Predictors include sociodemographics, psychosocial 
factors, and mistrust.

Inputs:
- Dataset must contain a column 'LCA_class' and variables listed in 'predictor_vars'.

Outputs:
- Excel file with regression coefficients, standard errors, p-values, and confidence intervals.

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_excel("gesundheitskompetenz_with_LCA_classes.xlsx")

# Define predictors and response
predictor_vars = [
    'GenderM0F1', 'AgeGroup', 'Education_cat', 'Language',
    'has_chronic_disease', 'PAM_level_cat', 'HLS_score_cat',
    'Mistrust_Index', 'Workinthehealthorsocialsector',
    'Lives_Alone_binary'
]

df_model = df[['LCA_class'] + predictor_vars].dropna()

# Convert LCA_class to numeric codes and predictors to dummies
df_model['LCA_class'] = pd.Categorical(df_model['LCA_class']).codes
X = pd.get_dummies(df_model[predictor_vars], drop_first=True)
y = df_model['LCA_class']

# Add constant
X = sm.add_constant(X)

# Fit multinomial logistic regression
mnlogit = sm.MNLogit(y, X)
result = mnlogit.fit(method='newton', maxiter=100, disp=False)

# Save output to Excel
summary_df = result.summary2().tables[1].reset_index()
summary_df.rename(columns={'index': 'Variable'}, inplace=True)
summary_df.to_excel("LCA_Multinomial_Regression_Unweighted_MNLogit.xlsx", index=False)

print("Model output saved to 'LCA_Multinomial_Regression_Unweighted_MNLogit.xlsx'")
