"""
* Noise correlations in A1 are behavior dependent (even after accounting for pupil / behavior gain changes?)
* Their state-dependence is correlated with behavioral performance, particularly in the "decision" window

Two figures:
    1) Overall change in noise correlation during 200ms decision window + correlation w/ behavior in that window
    2) Breakdown of delta noise correlation per time bin. Show where correlation strongest
"""
import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle('/auto/users/hellerc/code/projects/APAN2020/results/rsc_df.pickle')
df['diff'] = df['passive'] - df['active']
di_metric = 'DIref'
alpha = 1

# ======================== FIGURE 1 ================================
f, ax = plt.subplots(1, 2, figsize=(8, 4))

m1 = (df.batch == 307) & (df.tbin == '0.35_0.55') & ((df.pa < alpha) | (df.pp < alpha))
m2 = (df.batch.isin([302, 324])) & (df.tbin == '0.1_0.3') & ((df.pa < alpha) | (df.pp < alpha))
res = pd.concat([df[m1], df[m2]])

# paired barplot, one line per site, active / passive change in noise correlation
for s in res.site.unique():
    ax[0].plot([0, 1], 
                [res[res.site==s]['active'].mean(), res[res.site==s]['passive'].mean()],
                color='grey', alpha=1, lw=0.8)
resg = res.groupby(by='site').mean()
resgsem = res.groupby(by='site').sem()
ax[0].bar([0, 1], [resg['active'].mean(), resg['passive'].mean()], 
            yerr=[resg['active'].sem(), resg['passive'].sem()], edgecolor='k', lw=2)
ax[0].set_xlim((-2, 3))
ax[0].set_ylim((-0.05, 0.15))
ax[0].axhline(0, linestyle='--', color='k')
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['Active', 'Passive'], rotation=45)
ax[0].set_xlabel('Behavior State')
ax[0].set_ylabel(r'Noise Correlation ($r_{sc}$)')

# correlation with overall behavior
ax[1].scatter(resg[di_metric], resg['diff'], s=50, edgecolor='white')
ax[1].set_xlabel('Behavior performance (DI)')
ax[1].set_ylabel(r"$\Delta r_{sc}$"+"\n(Active - Passive)")
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.5, linestyle='--', color='k')
r, p = ss.pearsonr(resg[di_metric], resg['diff'])
ax[1].set_title(f"r: {round(r, 3)}, pval: {round(p, 3)}")

f.tight_layout()

plt.show()