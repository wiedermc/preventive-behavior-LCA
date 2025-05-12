# Preventive Behavior LCA

**Latent Class Analysis of Preventive Health Behavior Using GHP-16**

This repository contains the code, data, and outputs from a latent class analysis (LCA) study on preventive health behavior in a multilingual European population using the 16-item Good Health Practices (GHP-16) scale.

## Objectives
- Identify latent subgroups (classes) based on preventive health behavior patterns
- Examine how health literacy, patient activation, and sociocultural factors relate to behavioral classes
- Support targeted public health strategies through behavior-based segmentation

## Repository Contents
- `latent_class_analysis_ghp16.py`: Python script for performing LCA and generating outputs
- `gesundheitskompetenz_with_LCA_classes.xlsx`: Cleaned dataset with latent class labels
- `Table1_GHP16_LCA_Mean_SD.csv`: Summary table of GHP-16 means and standard deviations per class (Table 1 in manuscript)
- `Supplementary_Heatmap_GHP16_LCA.png`: Supplementary figure showing average behavior profiles by class across GHP-16 items

## Notes
- LCA was performed using Gaussian mixture modeling with standardized inputs
- The 5-class model was selected based on BIC and interpretability
- This project supports a broader effort to understand behavioral health heterogeneity
