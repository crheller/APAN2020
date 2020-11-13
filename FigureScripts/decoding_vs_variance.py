"""
decoding not corr with behavior, variance is
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
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 6

df = pd.read_pickle(DIR+"results/res.pickle")
df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
di_metric = 'DI'  # for this data, DI = DIref if df.aref_tar = True
dp_metric = 'dp_opt_sqrt'
df['lambda_tot'] = df['evals'].apply(lambda x: sum(x))
diff_norm = False
set_ylim = True

# ======================== FIGURE 1 ================================
f, ax = plt.subplots(1, 2, figsize=(4, 2))

mask = (df.aref_tar | df.aref_cat) & ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & (df.snr2!=-np.inf)
res = df[mask]
resga = res[res.active].groupby(by=['snr2', 'site']).mean()
resgp = res[~res.active].groupby(by=['snr2', 'site']).mean()

if diff_norm:
    diff = (resga['lambda_tot'] - resgp['lambda_tot']) / (resga['lambda_tot'] + resgp['lambda_tot'])
else:
    diff = resga['lambda_tot'] - resgp['lambda_tot']
#sns.regplot(x=resga[di_metric], y=diff, ax=ax[1])
ax[1].scatter(resga[di_metric], diff, s=25, edgecolor='white', color='tab:blue')
m, b = np.polyfit(resga[di_metric], diff, 1)
xran = np.linspace(ax[1].get_xlim()[0], ax[1].get_xlim()[1], 100)
ax[1].plot(xran, m * xran + b, lw=2, color='tab:blue')
ax[1].set_xlabel('Behavior performance (DI)')
ax[1].set_ylabel(r"$\Delta$ variance"+"\n(Active - Passive)")
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0.5, linestyle='--', color='grey')
r, p = ss.pearsonr(resga[di_metric], diff)
ax[1].set_title(r"$r$: %s, $p$: %s" % (round(r, 4), round(p, 4)))

# correlation with overall behavior
if diff_norm:
    diff = (resga[dp_metric] - resgp[dp_metric]) / (resga[dp_metric] + resgp[dp_metric])
else:
    diff = resga[dp_metric] - resgp[dp_metric]
ax[0].scatter(resga[di_metric], diff, s=25, edgecolor='white', color='tab:blue')
m, b = np.polyfit(resga[di_metric], diff, 1)
xran = np.linspace(ax[0].get_xlim()[0], ax[0].get_xlim()[1], 100)
ax[0].plot(xran, m * xran + b, lw=2, color='tab:blue')
#sns.regplot(x=resga[di_metric], y=diff, ax=ax[0])
ax[0].set_xlabel('Behavior performance (DI)')
ax[0].set_ylabel(r"$\Delta d'$"+"\n(Active - Passive)")
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].axvline(0.5, linestyle='--', color='grey')
r, p = ss.pearsonr(resga[di_metric], diff)
ax[0].set_title(r"$r$: %s, $p$: %s" % (round(r, 4), round(p, 4)))
if set_ylim:
    ax[0].set_ylim((None, 2))

f.tight_layout()

f.savefig(DIR + 'pyfigures/beh_dec_var.svg')

plt.show()