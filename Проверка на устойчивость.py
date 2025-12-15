import pandas as pd
import numpy as np
from scipy import stats

FILE = "nice_NOKAVKAZ_se.xlsx"
YEAR_COL = "year"
EDUC_COL = "educ"
GRP_COL = "grp"
Y_COL = "fin_share"

df = pd.read_excel(FILE)

df["grp_group"] = (
    df.groupby(YEAR_COL)[GRP_COL]
      .transform(lambda x: pd.qcut(
          x.rank(method="average"),
          3,
          labels=["low_GRP", "mid_GRP", "high_GRP"]
      ))
)

df["educ_group"] = (
    df.groupby(YEAR_COL)[EDUC_COL]
      .transform(lambda x: pd.qcut(
          x.rank(method="average"),
          3,
          labels=[1, 2, 3]
      ))
).astype(int)

results = []

for (year, grp_lvl), sub in df.groupby([YEAR_COL, "grp_group"]):
    sub = sub[[Y_COL, "educ_group"]].dropna()

    if sub["educ_group"].nunique() < 3:
        continue

    g1 = sub[sub["educ_group"] == 1][Y_COL]
    g2 = sub[sub["educ_group"] == 2][Y_COL]
    g3 = sub[sub["educ_group"] == 3][Y_COL]

    H, p = stats.kruskal(g1, g2, g3)

    N = len(sub)
    k = 3
    epsilon_sq = max((H - k + 1) / (N - k), 0)

    results.append({
        "year": year,
        "grp_group": grp_lvl,
        "H": H,
        "p_value": p,
        "epsilon_sq": epsilon_sq,
        "N": N
    })

robustness_df = pd.DataFrame(results).sort_values(
    ["year", "grp_group"]
)

print(robustness_df)