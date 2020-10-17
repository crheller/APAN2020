"""
Group across all targets. 

Show that tar vs. ref decoding gets better in active

Show relationship with behavior performance (mirroring the noise correlation figure)
"""
from settings import DIR
import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle(DIR+"results/res.pickle")
df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
di_metric = 'DI'  # for this data, DI = DIref if df.aref_tar = True
dp_metric = 'dp_opt_sqrt'
diff_norm = False

# ======================== FIGURE 1 ================================
f, ax = plt.subplots(1, 2, figsize=(8, 4))

mask = (df.aref_tar | df.aref_cat) & ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & (df.snr2!=-np.inf)
res = df[mask]

# paired barplot, one line per site, active / passive change in noise correlation
resga = res[res.active].groupby(by=['snr2', 'site']).mean()
resgp = res[~res.active].groupby(by=['snr2', 'site']).mean()
ax[0].scatter(resgp[dp_metric], resga[dp_metric], s=50, edgecolor='white', color='tab:blue')
mi = np.min(pd.concat([resga[dp_metric], resgp[dp_metric]]))
m = np.max(pd.concat([resga[dp_metric], resgp[dp_metric]]))
ax[0].plot([mi, m], [mi, m], linestyle='--', color='k')
ax[0].set_xlabel('Passive')
ax[0].set_ylabel('Active')
ax[0].set_title(r"Target vs. Reference Discriminability ($d'$)")

# correlation with overall behavior
if diff_norm:
    diff = (resga[dp_metric] - resgp[dp_metric]) / (resga[dp_metric] + resgp[dp_metric])
else:
    diff = resga[dp_metric] - resgp[dp_metric]
#ax[1].scatter(resga[di_metric], diff, s=50, edgecolor='white', color='tab:blue')
sns.regplot(x=resga[di_metric], y=diff, ax=ax[1])
ax[1].set_xlabel('Behavior performance (DI)')
ax[1].set_ylabel(r"$\Delta d'$"+"\n(Active - Passive)")
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.5, linestyle='--', color='k')
r, p = ss.pearsonr(resga[di_metric], diff)
ax[1].set_title(r"$r$: %s, $p$: %s" % (round(r, 3), round(p, 3)))

f.tight_layout()

f.savefig(DIR + 'pyfigures/beh_decoding.svg')

plt.show()