"""
VALIDATION: ALL-ELECTRIC SHARE vs HEAT VULNERABILITY
====================================================
Run after building_decarbonization_system.py, in the same session.

Updated for the corrected pipeline:
  has_heat_pump       -> all_electric
  hvi_score_building  -> hvi  (NaN where unmatched, no longer imputed to 3)
  'postal_code'       -> zip_col
  "NASA"              -> NYC DOHMH Heat Vulnerability Index

Reads the result honestly:
  - "All-electric" is a fuel-mix proxy. LL84 reports no equipment type, so this
    is not observed heat pump installation.
  - HVI is ordinal 1-5, so Spearman is the primary statistic. Pearson assumes an
    interval scale the data does not have.
  - HVI is assigned at ZIP level and correlates with income and housing age, so
    any relationship found here is associational and confounded.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

MIN_ZIP_SAMPLE = 10   # ZIPs below this produce unstable rates

# -----------------------------------------------------------------------------
# 1. Aggregate to ZIP level
# -----------------------------------------------------------------------------
rows = []
skipped_thin = skipped_unmatched = 0

for z, g in real_gdf.groupby(zip_col):
    if str(z).strip() in ("", "nan"):
        continue
    if not g["hvi_matched"].iloc[0]:
        skipped_unmatched += 1
        continue
    if len(g) < MIN_ZIP_SAMPLE:
        skipped_thin += 1
        continue
    rows.append({
        "zip": z,
        "hvi": int(g["hvi"].iloc[0]),
        "all_electric_pct": g["all_electric"].mean() * 100,
        "n_buildings": len(g),
    })

df_plot = pd.DataFrame(rows)

print(f"ZIPs analyzed        : {len(df_plot)}")
print(f"  excluded, no HVI   : {skipped_unmatched}")
print(f"  excluded, n < {MIN_ZIP_SAMPLE}   : {skipped_thin}")
print(f"Buildings represented: {int(df_plot['n_buildings'].sum()):,}")

if len(df_plot) < 5 or df_plot["hvi"].nunique() < 2:
    raise SystemExit("Not enough variation to test a relationship.")

# -----------------------------------------------------------------------------
# 2. Statistics
# -----------------------------------------------------------------------------
r, p_r = pearsonr(df_plot["hvi"], df_plot["all_electric_pct"])
rho, p_s = spearmanr(df_plot["hvi"], df_plot["all_electric_pct"])

# Building-weighted Pearson: a 500-building ZIP should not count the same as a
# 12-building ZIP. Reported alongside, not instead.
w = df_plot["n_buildings"]
x, y = df_plot["hvi"], df_plot["all_electric_pct"]
xm, ym = np.average(x, weights=w), np.average(y, weights=w)
cov = np.average((x - xm) * (y - ym), weights=w)
r_w = cov / np.sqrt(np.average((x - xm) ** 2, weights=w) * np.average((y - ym) ** 2, weights=w))

print(f"\nSpearman rho = {rho:+.3f}  (p = {p_s:.4f})   <- primary; HVI is ordinal")
print(f"Pearson  r   = {r:+.3f}  (p = {p_r:.4f})")
print(f"Pearson  r   = {r_w:+.3f}  (building-weighted)")

significant = p_s < 0.05
verdict = "SIGNIFICANT at p < 0.05" if significant else f"NOT significant (p = {p_s:.3f})"
print(f"\nVerdict: {verdict}")
if not significant:
    print("Do not present this as a validated relationship. Describe the tier")
    print("means below as an observed pattern, or leave the claim out entirely.")

tier_stats = (df_plot.groupby("hvi")["all_electric_pct"]
              .agg(mean="mean", median="median", n_zips="count").round(1))
tier_stats["buildings"] = df_plot.groupby("hvi")["n_buildings"].sum()
print("\nAll-electric share by heat vulnerability tier:")
print(tier_stats.to_string())

# -----------------------------------------------------------------------------
# 3. Figure
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.15]})

# Left: distribution per tier. With only five x-values, a scatter hides the
# spread that actually matters.
sns.boxplot(data=df_plot, x="hvi", y="all_electric_pct", ax=ax1,
            color="#dfe6e9", width=0.55, fliersize=0)
sns.stripplot(data=df_plot, x="hvi", y="all_electric_pct", ax=ax1,
              color="#0984e3", alpha=0.6, size=5, jitter=0.22)
ax1.set_title("Distribution by heat vulnerability tier", fontsize=12, fontweight="bold")
ax1.set_xlabel("NYC Heat Vulnerability Index (1 = lowest, 5 = highest)", fontsize=10)
ax1.set_ylabel("All-electric share of benchmarked buildings (%)", fontsize=10)

# Right: ZIP scatter sized by building count, with trend
sns.scatterplot(data=df_plot, x="hvi", y="all_electric_pct", size="n_buildings",
                sizes=(30, 400), alpha=0.65, color="#00b894",
                edgecolor="#00806a", linewidth=1.0, ax=ax2)
sns.regplot(data=df_plot, x="hvi", y="all_electric_pct", scatter=False, ax=ax2,
            color="#d63031", line_kws={"linestyle": "--", "linewidth": 2})
ax2.set_title(f"Spearman rho = {rho:+.3f} (p = {p_s:.3f}), n = {len(df_plot)} ZIPs",
              fontsize=12, fontweight="bold",
              color="#2d3436" if significant else "#b2bec3")
ax2.set_xlabel("NYC Heat Vulnerability Index (1 = lowest, 5 = highest)", fontsize=10)
ax2.set_ylabel("")
ax2.set_xticks([1, 2, 3, 4, 5])
ax2.legend(title="Buildings per ZIP", bbox_to_anchor=(1.02, 1), loc="upper left",
           frameon=True, fontsize=8, title_fontsize=9)

if not significant:
    ax2.text(0.5, 0.03, "Trend not statistically significant",
             transform=ax2.transAxes, ha="center", fontsize=9,
             color="#d63031", style="italic")

fig.suptitle(
    "All-Electric Building Share vs Neighborhood Heat Vulnerability, NYC",
    fontsize=14, fontweight="bold", y=1.00,
)
fig.text(0.5, -0.04,
         "All-electric inferred from LL84 fuel reporting (electricity present, no gas or oil), "
         "a proxy for electrification rather than observed heat pump installation.\n"
         "Heat vulnerability assigned at ZIP level (NYC DOHMH) and confounded with income "
         "and housing age. Associational, not causal.",
         ha="center", fontsize=8.5, color="#636e72")

plt.tight_layout()
plt.show()
