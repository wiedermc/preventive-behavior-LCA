
# table1_ghp16_lca.py
"""
Script to generate Table 1 for manuscript:
Mean (SD) scores of GHP-16 preventive health behavior items by latent class (5-class model).
Includes:
- Data loading
- Gaussian Mixture Model for LCA
- Summary table creation with mean and SD per item per class
- Export to CSV for inclusion in manuscript
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Load data
file_path = "gesundheitskompetenz_final_GHP16_PAM_levels_with_GHP16_total.xlsx"
df = pd.read_excel(file_path)

# Define GHP-16 rescaled item columns
ghp16_columns = [f'GHP16_{i}_rescal' for i in range(1, 17)]
ghp16_item_names = [
    "Exercise regularly", "Follow a balanced diet", "Take vitamins or dietary supplements",
    "Visit the dentist regularly", "Watch weight or try to lose weight",
    "Limit intake of high-fat or high-sugar foods", "Gather health information before making decisions",
    "Pay attention to physical symptoms or health changes", "Take supplements to prevent illness",
    "Get routine medical check-ups", "Floss teeth regularly", "Talk with others about health",
    "Don’t smoke", "Brush teeth twice a day", "Get vaccinated", "Get enough sleep"
]

# Subset and scale data
X = df[ghp16_columns].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit 5-class GMM for LCA
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)

# Attach labels to data
df_lca = df.loc[X.index].copy()
df_lca['LCA_class'] = labels + 1  # 1-based indexing

# Compute mean and SD by class
means = df_lca.groupby('LCA_class')[ghp16_columns].mean()
sds = df_lca.groupby('LCA_class')[ghp16_columns].std()
summary_table = means.round(2).astype(str) + " (" + sds.round(2).astype(str) + ")"
summary_table.index.name = "Latent Class"
summary_table = summary_table.T
summary_table.index = ghp16_item_names
summary_table.columns = [f"Class {i}" for i in summary_table.columns]

# Export to CSV
summary_table.to_csv("Table1_GHP16_LCA_Mean_SD.csv")

# Notes (for inclusion in manuscript)
notes = """
Table 1. Mean (SD) scores for each item of the 16-item Good Health Practices (GHP-16) scale,
stratified by latent class of preventive health behavior (n = XXXX).
Values represent means and standard deviations. Higher scores indicate more frequent engagement in the respective behavior.
Latent class membership was identified using Gaussian mixture modeling on rescaled GHP-16 items.
Class labels are for descriptive purposes only and do not imply rank or order.
"""
print(notes)
