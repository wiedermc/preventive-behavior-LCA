
# latent_class_analysis_ghp16.py

"""
Latent Class Analysis of Preventive Health Behavior using GHP-16 Items.
Includes:
- Reading Excel data
- Performing Gaussian Mixture-based LCA
- Selecting best model using BIC
- Assigning class labels
- Saving updated data and summary outputs
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "gesundheitskompetenz_final_GHP16_PAM_levels_with_GHP16_total.xlsx"
df = pd.read_excel(file_path)

# Define GHP-16 rescaled item columns
ghp16_columns = [f'GHP16_{i}_rescal' for i in range(1, 17)]
X = df[ghp16_columns].dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Gaussian Mixture Models for 2 to 6 classes
bic_scores, silhouette_scores, models = [], [], {}
for n_classes in range(2, 7):
    gmm = GaussianMixture(n_components=n_classes, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    models[n_classes] = (gmm, labels)

# Select best model
best_n = np.argmin(bic_scores) + 2
best_model, best_labels = models[best_n]
df_clusters = df.loc[X.index].copy()
df_clusters['LCA_class'] = best_labels + 1  # 1-based class labels

# Save updated dataset
df_clusters.to_excel("gesundheitskompetenz_with_LCA_classes.xlsx", index=False)

# Summary table: means and SDs
ghp_means = df_clusters.groupby('LCA_class')[ghp16_columns].mean()
ghp_stds = df_clusters.groupby('LCA_class')[ghp16_columns].std()
ghp_summary = ghp_means.round(2).astype(str) + " (" + ghp_stds.round(2).astype(str) + ")"
ghp_summary.T.to_csv("Table1_GHP16_LCA_Mean_SD.csv")

# Heatmap figure
class_profiles = ghp_means.T
ghp16_item_names = [
    "Exercise regularly", "Follow a balanced diet", "Take vitamins or dietary supplements",
    "Visit the dentist regularly", "Watch weight or try to lose weight",
    "Limit intake of high-fat or high-sugar foods", "Gather health information before making decisions",
    "Pay attention to physical symptoms or health changes", "Take supplements to prevent illness",
    "Get routine medical check-ups", "Floss teeth regularly", "Talk with others about health",
    "Don’t smoke", "Brush teeth twice a day", "Get vaccinated", "Get enough sleep"
]
class_profiles.index = [f"{i+1}. {name}" for i, name in enumerate(ghp16_item_names)]
class_profiles.columns = [f"Class {i}" for i in class_profiles.columns]

plt.figure(figsize=(12, 8))
sns.heatmap(class_profiles, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Mean Score'})
plt.title("Supplementary Figure: GHP-16 Behavioral Profiles by Latent Class")
plt.xlabel("Latent Class")
plt.ylabel("GHP-16 Item")
plt.tight_layout()
plt.savefig("Supplementary_Heatmap_GHP16_LCA.png", dpi=300)
plt.close()
