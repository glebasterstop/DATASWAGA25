import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import kruskal, rankdata, norm

df = pd.read_excel('2017_with_educ_lvl.xlsx')

educ_col = 'educ_lvl' if 'educ_lvl' in df.columns else [c for c in df.columns if 'educ' in c.lower()][0]
fin_col = 'fin_share' if 'fin_share' in df.columns else [c for c in df.columns if 'fin' in c.lower()][0]

df_filtered = df[df[educ_col].isin([1, 2, 3])].copy()

group1 = df_filtered[df_filtered[educ_col] == 1][fin_col].dropna().values
group2 = df_filtered[df_filtered[educ_col] == 2][fin_col].dropna().values
group3 = df_filtered[df_filtered[educ_col] == 3][fin_col].dropna().values

print(f"\nРазмеры выборок: n1={len(group1)}, n2={len(group2)}, n3={len(group3)}")

valid_groups = []
for g in [group1, group2, group3]:
    if len(g) > 0:
        valid_groups.append(g)

if len(valid_groups) >= 2:
    h_stat, p_kruskal = kruskal(*valid_groups)
else:
    p_kruskal = np.nan

dunn_results = []
if not np.isnan(p_kruskal) and p_kruskal < 0.05:
    groups = {1: group1, 2: group2, 3: group3}

    all_data = []
    labels = []
    for name, data in groups.items():
        if len(data) > 0:
            all_data.extend(data)
            labels.extend([name] * len(data))

    if len(all_data) > 0:
        ranks = rankdata(all_data)

        mean_ranks = {}
        for name in [1, 2, 3]:
            if len(groups[name]) > 0:
                mask = np.array(labels) == name
                mean_ranks[name] = np.mean(ranks[mask])

        pairs = [(1, 2), (1, 3), (2, 3)]
        n_comparisons = len(pairs)

        for g1_id, g2_id in pairs:
            if len(groups[g1_id]) > 0 and len(groups[g2_id]) > 0:
                n1, n2 = len(groups[g1_id]), len(groups[g2_id])
                N = len(all_data)

                se = np.sqrt((N * (N + 1) / 12) * (1 / n1 + 1 / n2))
                z = (mean_ranks[g1_id] - mean_ranks[g2_id]) / se
                p_raw = 2 * (1 - norm.cdf(abs(z)))
                p_bonferroni = p_raw

                dunn_results.append({
                    'g1': g1_id,
                    'g2': g2_id,
                    'p': p_bonferroni
                })

if not np.isnan(p_kruskal):
    print(f"\nКраскел-Уоллис: {p_kruskal:.6f}")
else:
    print(f"\nКраскел-Уоллис: N/A")

if dunn_results:
    print("\nДанн с поправкой Бонферрони:")
    for res in dunn_results:
        print(f"  {res['g1']} vs {res['g2']}: {res['p']:.6f}")
else:
    print("\nДанн с поправкой Бонферрони: не выполняется")