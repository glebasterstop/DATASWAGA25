import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene

df = pd.read_excel("nice_2016.xlsx")
if "year" in df.columns:
    df = df[df["year"] == 2016].copy()

if {"crime_people_econ", "crime_people"}.issubset(df.columns):
    denom = df["crime_people"].replace(0, np.nan)
    df["fin_share"] = df["crime_people_econ"] / denom
elif {"crime_rate_econ", "crime_rate"}.issubset(df.columns):
    denom = df["crime_rate"].replace(0, np.nan)
    df["fin_share"] = df["crime_rate_econ"] / denom
else:
    raise KeyError("Нужны либо crime_people_econ+crime_people, либо crime_rate_econ+crime_rate")

educ_candidates = [c for c in ["higher", "uni", "secondary_prof", "secondary_general", "general", "uneduc"] if c in df.columns]
if not educ_candidates:
    raise KeyError("Не найдены столбцы образования")
educ_base = "higher" if "higher" in df.columns else ("uni" if "uni" in df.columns else educ_candidates[0])

df = df.dropna(subset=[educ_base, "fin_share"]).copy()
df["educ_lvl"] = pd.qcut(df[educ_base], 3, labels=[1, 2, 3]).astype(int)

g1 = df.loc[df["educ_lvl"] == 1, "fin_share"].dropna().values
g2 = df.loc[df["educ_lvl"] == 2, "fin_share"].dropna().values
g3 = df.loc[df["educ_lvl"] == 3, "fin_share"].dropna().values

print(f"educ_base={educ_base}")
print(f"n1={len(g1)} n2={len(g2)} n3={len(g3)}")

for i, g in enumerate([g1, g2, g3], start=1):
    if len(g) < 3:
        print(f"Shapiro group {i}: NA")
    else:
        print(f"Shapiro group {i}: p={shapiro(g).pvalue:.6g}")

print(f"Levene (median): p={levene(g1, g2, g3, center='median').pvalue:.6g}")