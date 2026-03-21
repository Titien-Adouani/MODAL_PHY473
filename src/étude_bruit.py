import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import sys
 
# ── Lecture ───────────────────────────────────────────────────────────────────
path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Adouani\Desktop\MODAL_PHY473\nombre_pts_sde_fn_treshold.csv"
 
df = pd.read_csv(path, skipinitialspace=True)
df.columns = df.columns.str.strip()
 
if 'etage' not in df.columns:
    df['etage'] = 'haut'
else:
    df['etage'] = df['etage'].str.strip().fillna('haut')
 
df = df.dropna(subset=['HT', 'treshold', 'nb_sur_10000ms'])
df['HT']             = df['HT'].astype(int)
df['treshold']       = df['treshold'].astype(float)
df['nb_sur_10000ms'] = df['nb_sur_10000ms'].astype(float)
df['treshold']       = (-df['treshold'] * 10).round(4)
 
# ── Config ────────────────────────────────────────────────────────────────────
PALETTE = ['#38bdf8', '#f472b6', '#34d399', '#fb923c',
           '#a78bfa', '#facc15', '#f87171', '#818cf8']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
BG    = '#0f1117'
AX_BG = '#1a1d27'
GRID  = '#2d3748'
 
etages   = sorted(df['etage'].unique())
n_etages = len(etages)
 
# ── Une figure : lignes = étages, colonnes = linéaire | log ──────────────────
fig, axes = plt.subplots(n_etages, 2,
                         figsize=(14, 5 * n_etages),
                         squeeze=False)
fig.patch.set_facecolor(BG)
fig.suptitle('Taux de comptage = f(seuil)  —  par étage & HT',
             color='white', fontsize=14, fontweight='bold')
 
grp = (df.groupby(['etage', 'HT', 'treshold'])['nb_sur_10000ms']
         .agg(['mean', 'std']).reset_index())
grp.columns = ['etage', 'HT', 'treshold', 'mean', 'std']
 
for row, etage in enumerate(etages):
    sub_etage = grp[grp['etage'] == etage]
    HT_vals   = sorted(sub_etage['HT'].unique())
 
    for col, logy in enumerate([False, True]):
        ax = axes[row][col]
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors='#a0aec0', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, which='both', color=GRID, linewidth=0.6, linestyle='--')
        ax.set_xlabel('Seuil (× 10 mV)', color='#a0aec0', fontsize=10)
        ax.set_ylabel('Comptage / 10 s',  color='#a0aec0', fontsize=10)
        ax.set_title(f'Étage : {etage}  —  {"log" if logy else "linéaire"}',
                     color='white', fontsize=11, fontweight='bold')
        if logy:
            ax.set_yscale('log')
 
        for i, ht in enumerate(HT_vals):
            sub = sub_etage[sub_etage['HT'] == ht].sort_values('treshold')
            color  = PALETTE[i % len(PALETTE)]
            marker = MARKERS[i % len(MARKERS)]
            ax.plot(sub['treshold'], sub['mean'],
                    marker=marker, markersize=5, linewidth=2,
                    color=color, label=f'HT = {ht} V',
                    markerfacecolor='white', markeredgewidth=1.2)
            lo = (sub['mean'] - sub['std']).clip(lower=0.1) if logy else sub['mean'] - sub['std']
            ax.fill_between(sub['treshold'], lo, sub['mean'] + sub['std'],
                            alpha=0.12, color=color)
 
        if col == 0:
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor='white', fontsize=8)
 
plt.tight_layout()
out = path.rsplit('.', 1)[0] + '_graph.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"Sauvegardé : {out}")
plt.show()
