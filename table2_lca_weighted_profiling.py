
# table2_lca_weighted_profiling.py

"""
Generates Table 2: Weighted descriptive profiling of latent classes (LCA) by sociodemographic and behavioral factors.
Outputs:
- Formatted Word document with weighted % (unweighted counts)
- Structured Excel file for GitHub and manuscript use

Data:
- Input: Excel file with GHP-16 items, weight column ('Gewicht'), and profiling variables

Steps:
- Gaussian mixture modeling for 5-class LCA (unweighted)
- Apply weights post hoc to calculate weighted proportions
- Compute chi-square and Cramér’s V based on weighted counts
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2_contingency
from docx import Document

# Load input data
df = pd.read_excel("gesundheitskompetenz_final_GHP16_PAM_levels_with_GHP16_total.xlsx")

# Define variables
ghp16_columns = [f"GHP16_{i}_rescal" for i in range(1, 17)]
class_labels = {
    1: "1. Broadly Moderate",
    2: "2. Globally Low",
    3: "3. Medically Passive",
    4: "4. Peripherally Engaged",
    5: "5. High Engagers"
}

# Ensure weight column exists
df["Gewicht"] = df["Gewicht"].fillna(1.0)

# Run LCA (Gaussian Mixture)
X = df[ghp16_columns].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)
df_profile = df.loc[X.index].copy()
df_profile["LCA_class"] = labels + 1
df_profile["LanguageSimple"] = df_profile["Language"].replace({"D": "German", "I": "Italian"})
df_profile["LanguageSimple"] = df_profile["LanguageSimple"].where(
    df_profile["LanguageSimple"].isin(["German", "Italian"]), "Other"
)

# Label maps
value_labels = {
    'GenderM0F1': {'0': 'Female', '1': 'Male'},
    'AgeGroup': {'18-34': '18–34', '35-54': '35–54', '55-99': '55–99'},
    'Education_cat': {'1': 'Middle school', '2': 'Vocational school', '3': 'High school', '4': 'University'},
    'LanguageSimple': {'German': 'German', 'Italian': 'Italian', 'Other': 'Other'},
    'HLS_score_cat': {'0': 'Inadequate', '1': 'Problematic', '2': 'Sufficient', 'Missing': 'Missing'},
    'PAM_level_cat': {
        '1': 'Disengaged and overwhelmed',
        '2': 'Becoming aware but still struggling',
        '3': 'Taking action',
        '4': 'Maintaining behaviors and pushing further'
    },
    'has_chronic_disease': {'0': 'No', '1': 'Yes'}
}

# Export structured table to Excel
rows = []
for var, labels in value_labels.items():
    df_profile[var] = df_profile[var].astype(str)
    keys = list(labels.keys())
    display = list(labels.values())
    weighted_ct = pd.crosstab(df_profile['LCA_class'], df_profile[var],
                              values=df_profile['Gewicht'], aggfunc='sum').fillna(0)[keys]
    weighted_pct = weighted_ct.div(weighted_ct.sum(axis=1), axis=0).round(1)
    unweighted_ct = pd.crosstab(df_profile['LCA_class'], df_profile[var])[keys]

    for i, class_name in class_labels.items():
        row = {"Variable": var, "Class": class_name}
        for j, key in enumerate(keys):
            row[labels[key]] = f"{weighted_pct.loc[i, key]} ({unweighted_ct.loc[i, key]})"
        rows.append(row)

df_output = pd.DataFrame(rows)
df_output.to_excel("Table2_LCA_Class_Profiling_Weighted.xlsx", index=False)

print("Saved Table 2 Excel file with weighted descriptive profiling.")
