"""
Compare pairwise noise correlations to changes along "noise correlation axis".

Main Q: is delta variance on PC_1 correlated with change in mean pairwise rsc? Think Ni et al 2018 --
    Idea is for this to be an early figure that says, "forget pairwise correlations, think about this 
    as a low D change in variance in state space"
"""
import helpers
from settings import DIR
import scipy.stats as ss
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

rsc = pd.read_pickle(DIR + 'results/rsc_df.pickle')
lv = pickle.load(open(DIR + 'results/drsc_axes.pickle', "rb" ))

di_metric='DIref'
sigcorr = False
alpha = 0.05
 # time window used for lv estimation should match time bin uses for rsc calculation
tbins = {
    302: '0.1_0.3',
    307: '0.35_0.55',
    324: '0.1_0.3',
    325: '0.1_0.3' 
}
# correlation between eval1 and delta pairswise noise correlation across sites
f, ax = plt.subplots(1, 1, figsize=(4, 4))
x = []
y = []
beh = []
sig = []
for site in rsc.site.unique():
    if rsc[rsc.site==site].batch.iloc[0]==307:
        tbin = tbins[307]
    elif rsc[rsc.site==site].batch.iloc[0]==302:
        tbin = tbins[302]
    elif rsc[rsc.site==site].batch.iloc[0]==324:
        tbin = tbins[324]
    elif rsc[rsc.site==site].batch.iloc[0]==325:
        tbin = tbins[325]

    if sigcorr:
        sigmask = (rsc.pp < alpha) | (rsc.pa < alpha)
        diff = (rsc[(rsc.site==site) & (rsc.tbin==tbin) & sigmask]['passive'] - rsc[(rsc.site==site) & (rsc.tbin==tbin) & sigmask]['active']).mean()
    else:
        diff = (rsc[(rsc.site==site) & (rsc.tbin==tbin)]['passive'] - rsc[(rsc.site==site) & (rsc.tbin==tbin)]['active']).mean()
    if lv[site]['tarOnly']['nSigDim'] >= 1:
        sig.append(1)
    else:
        sig.append(0)
    x.append(diff)
    lv_delta = (lv[site]['tarOnly']['evals'][0]**2) / sum(abs(lv[site]['tarOnly']['evals'])**2)
    y.append(lv_delta)
    rsc.at[(rsc.site==site) & (rsc.tbin==tbin), 'lv_delta'] = lv_delta
    mb = rsc[(rsc.site==site) & (rsc.tbin==tbin)].groupby('snr').mean()[di_metric]
    beh.append(mb[mb!=np.inf].mean())

ax.scatter(x, y, s=50, edgecolor='white', color='k')
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.set_xlabel(r"$\Delta r_{sc}$")
ax.set_ylabel(r"$\Delta$ variance"+"\non noise correlation axis")

X = pd.DataFrame(data=x, columns=['rsc'])
X = sm.add_constant(X)
y = pd.Series(data=y)
r = helpers.fit_OLS_model(X, y, replace=False, nboot=100, njacks=20)
ax.set_title(r"$R^2: %s, p < 0.05$" % round(r['r2']['rsc'], 3))

f.tight_layout()

f.savefig(DIR + 'pyfigures/rsc_axis_vs_pairwise.svg')

plt.show()