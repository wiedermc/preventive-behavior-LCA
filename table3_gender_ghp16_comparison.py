
# table3_gender_ghp16_comparison.py

"""
Script to generate Table 3: Population-based comparison of GHP-16 preventive health behavior items between women and men.

Outputs:
- Weighted means and SDs for each item by gender
- Unweighted Mann–Whitney U test p-values
- Cliff’s delta effect sizes
- Difference in weighted means (Male - Female)

Data Requirements:
- Input file: 'gesundheitskompetenz_final_GHP16_PAM_levels_with_GHP16_total.xlsx'
- Required columns: 'GenderM0F1', 'Gewicht', 'GHP16_1_rescal' to 'GHP16_16_rescal'
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.weightstats import DescrStatsW

# Load data
df = pd.read_excel("gesundheitskompetenz_final_GHP16_PAM_levels_with_GHP16_total.xlsx")

# Define columns
gender_col = 'GenderM0F1'
weight_col = 'Gewicht'
ghp16_items = [f"GHP16_{i}_rescal" for i in range(1, 17)]

# Clean data
df_clean = df[[gender_col, weight_col] + ghp16_items].dropna()

# Cliff's delta function
def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    n_tot = nx * ny
    more = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)
    return (more - less) / n_tot if n_tot > 0 else np.nan

# Run comparison
results = []
for item in ghp16_items:
    female = df_clean[df_clean[gender_col] == 0]
    male = df_clean[df_clean[gender_col] == 1]

    female_stats = DescrStatsW(female[item], weights=female[weight_col], ddof=0)
    male_stats = DescrStatsW(male[item], weights=male[weight_col], ddof=0)

    mean_f, sd_f = female_stats.mean, female_stats.std
    mean_m, sd_m = male_stats.mean, male_stats.std
    mean_diff = mean_m - mean_f

    # Mann–Whitney U
    stat, p_val = mannwhitneyu(female[item], male[item], alternative='two-sided')
    delta = cliffs_delta(female[item].tolist(), male[item].tolist())

    results.append({
        'Item': item,
        'Weighted Mean (SD) – Female': f"{mean_f:.2f} ({sd_f:.2f})",
        'Weighted Mean (SD) – Male': f"{mean_m:.2f} ({sd_m:.2f})",
        'Mann–Whitney p-value': f"{p_val:.4f}",
        "Cliff's delta": f"{delta:.2f}",
        'Weighted Mean Difference (M-F)': f"{mean_diff:.2f}"
    })

# Export
df_out = pd.DataFrame(results)
df_out.to_excel("GHP16_Gender_Comparison_FullStats.xlsx", index=False)
print("Saved Table 3 to GHP16_Gender_Comparison_FullStats.xlsx")
