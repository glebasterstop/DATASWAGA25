import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.stats import rankdata, norm

# ===== НАСТРОЙКИ =====
FILE = "nice_2020.xlsx"
EDUC_COL = "educ"        # по чему делим на группы
Y_COL = "crime_rate"    # что сравниваем
# ====================

def dunn_posthoc_bonferroni(values, groups):
    df = pd.DataFrame({"val": values, "grp": groups}).dropna()
    df["rank"] = rankdata(df["val"])
    N = len(df)

    uniq, counts = np.unique(df["val"], return_counts=True)
    tie_sum = np.sum(counts**3 - counts)
    tie_corr = 1 - tie_sum / (N**3 - N)

    const = (N * (N + 1) / 12) * tie_corr

    res = []
    grps = sorted(df["grp"].unique())
    for i in range(len(grps)):
        for j in range(i + 1, len(grps)):
            g1, g2 = grps[i], grps[j]
            r1 = df[df["grp"] == g1]["rank"]
            r2 = df[df["grp"] == g2]["rank"]

            z = (r1.mean() - r2.mean()) / np.sqrt(const * (1/len(r1) + 1/len(r2)))
            p = 2 * (1 - norm.cdf(abs(z)))
            res.append((g1, g2, z, p))

    m = len(res)
    out = []
    for g1, g2, z, p in res:
        out.append((g1, g2, z, p, min(p * m, 1.0), abs(z) / np.sqrt(N)))

    return pd.DataFrame(
        out,
        columns=["group1", "group2", "z", "p", "p_bonf", "effect_r"]
    )


# ===== ОСНОВНОЙ РАСЧЁТ =====
df = pd.read_excel(FILE)
df = df[[EDUC_COL, Y_COL]].dropna()

# 3 равные группы по educ
df["educ_group"] = pd.qcut(
    df[EDUC_COL].rank(method="average"),
    3,
    labels=[1, 2, 3]
).astype(int)

# описательная статистика
desc = df.groupby("educ_group")[Y_COL].agg(
    n="count",
    mean="mean",
    median="median",
    std="std"
)

# Shapiro–Wilk
shapiro = (
    df.groupby("educ_group")[Y_COL]
      .apply(lambda x: pd.Series(stats.shapiro(x), index=["W", "p"]))
)

# Levene
g1 = df[df["educ_group"] == 1][Y_COL]
g2 = df[df["educ_group"] == 2][Y_COL]
g3 = df[df["educ_group"] == 3][Y_COL]
levene_stat, levene_p = stats.levene(g1, g2, g3, center="median")

# Kruskal–Wallis + эффект
H, p_kw = stats.kruskal(g1, g2, g3)
N = len(df)
k = 3
epsilon_sq = (H - k + 1) / (N - k)

# Dunn + Bonferroni
dunn = dunn_posthoc_bonferroni(df[Y_COL].values, df["educ_group"].values)

# ===== ВЫВОД ЧИСТО ЧИСЕЛ =====
print("\nDESCRIPTIVE")
print(desc)

print("\nSHAPIRO")
print(shapiro)

print("\nLEVENE")
print({"stat": levene_stat, "p": levene_p})

print("\nKRUSKAL")
print({"H": H, "p": p_kw, "epsilon_sq": epsilon_sq})

print("\nDUNN (BONFERRONI)")
print(dunn)