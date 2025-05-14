
# vif_and_correlation_analysis.py

"""
Script for Multicollinearity Diagnostics:
- Computes Variance Inflation Factors (VIF) for all predictors in a multinomial logistic regression
- Computes a Pearson correlation matrix of all predictors

Inputs:
- Requires a dataset with clean predictor variables and optional weights
- Drops NA values for complete case analysis

Outputs:
- Excel file with VIF values and correlation matrix in separate sheets

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_excel("gesundheitskompetenz_with_LCA_classes.xlsx")

# Define model variables
model_vars = [
    'LCA_class', 'GenderM0F1', 'AgeGroup', 'Education_cat', 'Language',
    'has_chronic_disease', 'PAM_level_cat', 'HLS_score_cat',
    'Mistrust_Index', 'Workinthehealthorsocialsector',
    'Lives_Alone_binary', 'Gewicht'
]

# Drop missing values
df_model = df[model_vars].dropna()

# Drop LCA_class and weight for multicollinearity checks
X = pd.get_dummies(df_model.drop(columns=['LCA_class', 'Gewicht']), drop_first=True)

# Add constant for VIF calculation
X_const = sm.add_constant(X)

# Calculate VIFs
vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

# Calculate Pearson correlation matrix
correlation_matrix = X.corr().round(2)

# Export to Excel
with pd.ExcelWriter("LCA_Predictor_VIF_and_Correlation.xlsx") as writer:
    vif_data.to_excel(writer, sheet_name="VIF", index=False)
    correlation_matrix.to_excel(writer, sheet_name="Correlation_Matrix")

print("VIF and correlation matrix exported to 'LCA_Predictor_VIF_and_Correlation.xlsx'")
