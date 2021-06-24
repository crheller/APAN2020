"""
Group across all targets. 

Show that tar vs. ref decoding gets better in active

Show relationship with behavior performance (mirroring the noise correlation figure)
"""
import helpers
import statsmodels.api as sm
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob
from settings import DIR
import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

df = pd.read_pickle(DIR+"results/res_pr.pickle")
df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
di_metric = 'DI'  # for this data, DI = DIref if df.aref_tar = True
dp_metric = 'dp_opt_sqrt'
diff_norm = False
set_ylim = True

# ======================== FIGURE 1 ================================
f, ax = plt.subplots(1, 2, figsize=(4, 2))

mask = (df.aref_tar | df.aref_cat) & ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & (df.snr2!=-np.inf)
res = df[mask]

# paired barplot, one line per site, active / passive change in noise correlation
resga = res[res.active].groupby(by=['snr2', 'site']).mean()
resgp = res[~res.active].groupby(by=['snr2', 'site']).mean()
ax[0].scatter(resgp[dp_metric], resga[dp_metric], s=10, edgecolor='white', color='tab:blue')
mi = np.min(pd.concat([resga[dp_metric], resgp[dp_metric]]))
m = np.max(pd.concat([resga[dp_metric], resgp[dp_metric]]))
ax[0].plot([mi, m], [mi, m], linestyle='--', color='k')
ax[0].set_xlabel('Passive')
ax[0].set_ylabel('Active')

d = {s: resga.loc[pd.IndexSlice[:, str(s)], dp_metric].values - 
                    resgp.loc[pd.IndexSlice[:, str(s)], dp_metric].values for s in resga.index.get_level_values(1).unique()}
bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
p = get_direct_prob(bootsamp, np.zeros(len(bootsamp)))[0]
print(r"Target vs. Reference Discriminability ($d'$)"+
        f"\n active: {round(resga[dp_metric].mean().astype(float), 3)}, passive: {round(resgp[dp_metric].mean().astype(float), 3)}, pval: {round(p, 3)}")

# correlation with overall behavior
if diff_norm:
    diff = (resga[dp_metric] - resgp[dp_metric]) / (resga[dp_metric] + resgp[dp_metric])
else:
    diff = resga[dp_metric] - resgp[dp_metric]
#ax[1].scatter(resga[di_metric], diff, s=50, edgecolor='white', color='tab:blue')
sns.regplot(x=resga[di_metric], y=diff, ax=ax[1], scatter_kws={'s': 10, 'edgecolor': 'white'})
ax[1].set_xlabel('Behavior performance (DI)')
ax[1].set_ylabel(r"$\Delta d'$"+"\n(Active - Passive)")
ax[1].axhline(0, linestyle='--', color='k')
ax[1].axvline(0.5, linestyle='--', color='k')
'''
r, p = ss.pearsonr(resga[di_metric], diff)
ax[1].set_title(r"$r$: %s, $p$: %s" % (round(r, 3), round(p, 3)))
'''
# test correlation with cross validated regression
X = pd.concat([diff], axis=1)
X -= X.mean(axis=0)
X /= X.std(axis=0)
X = sm.add_constant(X)
y = resga[di_metric]
results = helpers.fit_OLS_model(X, y, replace=False, nboot=100, njacks=5)
print(r"$cvR^2$: %s, slope: %s, $95 CI$: %s %s" % (round(results['r2'][diff.name], 3), round(results['coef'][diff.name], 3), 
                                round(results['ci_coef'][diff.name][0], 3),  round(results['ci_coef'][diff.name][1], 3)))

if set_ylim:
    ax[1].set_ylim((None, 2))
    ax[1].set_xlim((0.4, 1))
    # add outlier
    g = np.argwhere(diff.values>2).squeeze()
    ax[1].scatter(resga[di_metric].iloc[g], 2-0.1, color='r', s=10)
    ax[1].text(resga[di_metric].iloc[g]+0.05, 2-0.1, f"({round(float(resga[di_metric].iloc[g]), 3)}, {round(float(diff.iloc[g]), 3)})", fontsize=4)
f.tight_layout()

f.savefig(DIR + 'pyfigures/beh_decoding.svg')

plt.show()