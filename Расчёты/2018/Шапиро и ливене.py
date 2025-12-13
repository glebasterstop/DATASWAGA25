import pandas as pd
from scipy.stats import shapiro, levene

df = pd.read_excel('2018_with_educ_lvl.xlsx')

educ_col = 'educ_lvl'
fin_col = 'fin_share'

g1 = df[df[educ_col] == 1][fin_col].dropna().values
g2 = df[df[educ_col] == 2][fin_col].dropna().values
g3 = df[df[educ_col] == 3][fin_col].dropna().values

print("Шапиро:")
print(f"Группа 1: {shapiro(g1)[1]:.6f}")
print(f"Группа 2: {shapiro(g2)[1]:.6f}")
print(f"Группа 3: {shapiro(g3)[1]:.6f}")

print(f"\nЛевене: {levene(g1, g2, g3)[1]:.6f}")