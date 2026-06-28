# ==============================================================================
# GEOSPATIAL VALIDATION ENGINE - REGENERATE ADOPTION VS THERMAL MATRIX
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

print("📊 Extracting live neighborhood layers for statistical verification...")

# 1. Aggregate the active dataframe by ZIP code to find true adoption metrics
validation_set = []
for z_code, group in real_gdf.groupby('postal_code'):
    if z_code == "nan" or str(z_code).strip() == "":
        continue

    total_buildings = len(group)
    electrified_count = int(group['has_heat_pump'].sum())

    # Calculate the localized adoption percentage
    adoption_rate = (electrified_count / total_buildings) * 100 if total_buildings > 0 else 0
    nasa_heat_tier = int(group['hvi_score_building'].iloc[0])

    validation_set.append({
        'ZIP Code': z_code,
        'NASA Heat Index': nasa_heat_tier,
        'Commercial Heat Pump Adoption Rate (%)': adoption_rate,
        'Total Commercial Buildings Checked': total_buildings
    })

df_plot = pd.DataFrame(validation_set)

# 2. Compute the exact live Pearson correlation coefficient
r_val, p_val = pearsonr(df_plot['NASA Heat Index'], df_plot['Commercial Heat Pump Adoption Rate (%)'])
print(f"🎯 Live Verification Complete! Pearson r = {r_val:.3f} (p-value: {p_val:.4f})")

# 3. Render the production-grade visual matrix
plt.figure(figsize=(9, 6.5))
sns.set_theme(style="ticks")

# Plot individual neighborhood data bubbles scaled by building density volume
sns.scatterplot(
    data=df_plot,
    x='NASA Heat Index',
    y='Commercial Heat Pump Adoption Rate (%)',
    size='Total Commercial Buildings Checked',
    sizes=(40, 400),
    alpha=0.7,
    color='#2ecc71',
    edgecolor='#27ae60',
    linewidth=1.2
)

# Overlay the linear regression trend line to show capital misallocation paths
sns.regplot(
    data=df_plot,
    x='NASA Heat Index',
    y='Commercial Heat Pump Adoption Rate (%)',
    scatter=False,
    color='#e74c3c',
    line_kws={"linestyle": "--", "linewidth": 2}
)

# Formatting bounds and clean typography labels
plt.title(f"Commercial Heat Pump Adoption vs. NASA Satellite Thermal Index\n(Pearson r = {r_val:.3f})",
          fontsize=13, fontweight='bold', pad=15, fontname='sans-serif')
plt.xlabel("NASA Climate Vulnerability / Heat Island Index (1 = Cooler, 5 = Hottest)", fontsize=11, labelpad=10)
plt.ylabel("Commercial Heat Pump Adoption Rate (%)", fontsize=11, labelpad=10)

plt.xticks([1, 2, 3, 4, 5])
plt.grid(True, linestyle=':', alpha=0.5)

# Style and position the density volume legend
plt.legend(title="Total Buildings per ZIP", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()

# Render directly into your active Colab cell interface window
plt.show()
