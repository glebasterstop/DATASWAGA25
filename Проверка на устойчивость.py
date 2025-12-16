import pandas as pd
import numpy as np
from scipy import stats

# -----------------------
# SETTINGS
# -----------------------
FILE = "nice_NOKAVKAZ_se.xlsx"
YEAR_COL = "year"
EDUC_COL = "educ"
GRP_COL  = "grp"
Y_COL    = "fin_share"

# -----------------------
# LOAD
# -----------------------
df = pd.read_excel(FILE)

# safety: numeric outcome
df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

# -----------------------
# 1) GRP groups (within each year): low/mid/high GRP
# -----------------------
def tertile_labels_grp(x: pd.Series) -> pd.Series:
    # rank to reduce issues with ties; method="first" avoids non-unique bin edges
    r = x.rank(method="first")
    return pd.qcut(r, 3, labels=["low_GRP", "mid_GRP", "high_GRP"])

df["grp_group"] = df.groupby(YEAR_COL)[GRP_COL].transform(tertile_labels_grp)

# -----------------------
# 2) EDUC groups (within each year AND within each GRP group):
#    1/2/3 = low/mid/high education
# -----------------------
def tertile_labels_educ(x: pd.Series) -> pd.Series:
    r = x.rank(method="first")
    return pd.qcut(r, 3, labels=[1, 2, 3])

df["educ_group"] = (
    df.groupby([YEAR_COL, "grp_group"])[EDUC_COL]
      .transform(tertile_labels_educ)
)

# convert to int safely (keep NA)
df["educ_group"] = pd.to_numeric(df["educ_group"], errors="coerce").astype("Int64")

# -----------------------
# 3) Kruskal–Wallis within each (year, grp_group)
# -----------------------
results = []

for (year, grp_lvl), sub in df.groupby([YEAR_COL, "grp_group"], dropna=True):
    sub = sub[[Y_COL, "educ_group"]].dropna()

    # Split into 3 education groups
    g1 = sub.loc[sub["educ_group"] == 1, Y_COL].astype(float)
    g2 = sub.loc[sub["educ_group"] == 2, Y_COL].astype(float)
    g3 = sub.loc[sub["educ_group"] == 3, Y_COL].astype(float)

    # Need all 3 groups present and not too tiny
    if min(len(g1), len(g2), len(g3)) < 2:
        continue

    H, p = stats.kruskal(g1, g2, g3, nan_policy="omit")

    N = len(sub)
    k = 3

    # epsilon-squared (effect size) with guard
    epsilon_sq = max((H - k + 1) / (N - k), 0) if N > k else np.nan

    results.append({
        "year": int(year),
        "grp_group": str(grp_lvl),
        "H": float(H),
        "p_value": float(p),
        "epsilon_sq": float(epsilon_sq) if pd.notna(epsilon_sq) else np.nan,
        "N": int(N),
        "n_g1": int(len(g1)),
        "n_g2": int(len(g2)),
        "n_g3": int(len(g3)),
    })

robustness_df = (
    pd.DataFrame(results)
      .sort_values(["year", "grp_group"])
      .reset_index(drop=True)
)

print(robustness_df)

# Optional: save
# robustness_df.to_excel("robustness_by_year_by_grp.xlsx", index=False)