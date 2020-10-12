"""
Compare pairwise noise correlations to changes along "noise correlation axis".

Main Q: is delta variance on PC_1 correlated with change in mean pairwise rsc? Think Ni et al 2018 --
    Idea is for this to be an early figure that says, "forget pairwise correlations, think about this 
    as a low D change in variance in state space"
"""
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

rsc = pd.read_pickle('/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/rsc_df.pickle')
lv = pickle.load(open('/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/drsc_axes.pickle', "rb" ))

tbin = '0.1_0.3'  # only time window used for lv estimation so far

# correlation between eval1 and delta pairswise noise correlation across sites
f, ax = plt.subplots(1, 1, figsize=(5, 5))
x = []
y = []
sig = []
for site in lv.keys():
    diff = (rsc[(rsc.site==site) & (rsc.tbin==tbin)]['passive'] - rsc[(rsc.site==site) & (rsc.tbin==tbin)]['active']).mean()
    if lv[site]['tarCat']['nSigDim'] >= 1:
        sig.append(1)
    else:
        sig.append(0)
    x.append(diff)
    y.append(lv[site]['tarCat']['evals'][0] / sum(abs(lv[site]['tarCat']['evals'])))

ax.scatter(x, y, s=50, edgecolor='k', color='tab:blue')
ax.scatter(np.array(x)[np.array(sig)==1], 
                np.array(y)[np.array(sig)==1], s=50, edgecolor='k', color='tab:orange')
ax.set_xlabel(r"$\Delta r_{sc}$")
ax.set_ylabel(r"$\Delta$ variance"+"\non noise corr. axis")

f.tight_layout()

plt.show()
