"""
Exploratory analysis of pairwise rsc
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

df = pd.read_pickle(DIR+'/results/rsc_df.pickle')
df['diff'] = df['passive'] - df['active']
tbin = ['0_0.1', '0.1_0.2', '0.2_0.3', '0.3_0.4', '0.4_0.5']
di = 'DIref'
# plot active/passive as fn of time bin
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

x = np.arange(len(tbin))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax[0].bar(x - width/2, df[df.area=='A1'].groupby(by='tbin').mean()['active'].loc[tbin], width, 
                             yerr=df[df.area=='A1'].groupby(by='tbin').sem()['active'].loc[tbin], label='Active')
rects2 = ax[0].bar(x + width/2, df[df.area=='A1'].groupby(by='tbin').mean()['passive'].loc[tbin], width, 
                             yerr=df[df.area=='A1'].groupby(by='tbin').sem()['passive'].loc[tbin], label='Passive')
ax[0].set_xticks(x)
ax[0].set_xticklabels(tbin)
ax[0].legend(frameon=False)

ax[0].set_xlabel("Time Window")
ax[0].set_ylabel(r"$r_{sc}$")
ax[0].set_title('A1')

rects1 = ax[1].bar(x - width/2, df[df.area=='PEG'].groupby(by='tbin').mean()['active'].loc[tbin], width, 
                             yerr=df[df.area=='PEG'].groupby(by='tbin').sem()['active'].loc[tbin], label='Active')
rects2 = ax[1].bar(x + width/2, df[df.area=='PEG'].groupby(by='tbin').mean()['passive'].loc[tbin], width, 
                             yerr=df[df.area=='PEG'].groupby(by='tbin').sem()['passive'].loc[tbin], label='Passive')
ax[1].set_xticks(x)
ax[1].set_xticklabels(tbin)
ax[1].legend(frameon=False)

ax[1].set_xlabel("Time Window")
ax[1].set_ylabel(r"$r_{sc}$")
ax[1].set_title('PEG')

f.tight_layout()

# recapitulate Cosyne plot?? -- plot delta rsc as fn of behavior performance (on average across targets)
di = 'DIref'
mask = (df.batch==307)
tbin307 = ['0.15_0.25', '0.25_0.35', '0.35_0.45', '0.45_0.55', '0.55_0.65', '0.65_0.75']
f, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True)
for a, tb in zip(ax.flatten(), tbin307):
    diff = df[mask & (df.tbin==tb) & ((df.pa<0.05) | (df.pp<0.05))].groupby(by='site').mean()['diff']
    b = df[mask & (df.tbin==tb) & ((df.pa<0.05) | (df.pp<0.05))].groupby(by='site').mean()[di]
    a.scatter(b, diff, s=40, edgecolor='k')
    a.set_xlabel('Behavior performance (DI)')
    a.set_ylabel(r"$\Delta r_{sc}$")
    a.axhline(0, linestyle='--', color='k')
    r, p = ss.pearsonr(b, diff)
    a.set_title(f"Time bin: {tb}\nr: {round(r, 3)}, pval:  {round(p, 3)}")
f.tight_layout()

mask = df.batch==307
tbin307 = ['0.15_0.25', '0.25_0.35', '0.35_0.45', '0.45_0.55', '0.55_0.65', '0.65_0.75']
f, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True)
for i, (a, tb) in enumerate(zip(ax.flatten(), tbin307)):
    diff = df[mask & (df.tbin==tb) & ((df.pa<0.05) | (df.pp<0.05)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()['diff']
    b = df[mask & (df.tbin==tb) & ((df.pa<0.05) | (df.pp<0.05)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()[di]
    snr = b.index.get_level_values(0)
    b = b[~diff.isna()].values
    snr = snr[~diff.isna()]
    diff = diff[~diff.isna()].values
    _df = pd.DataFrame(columns=['diff', 'di', 'snr'], data=np.stack([diff, b, snr]).T)
    _df = _df.astype({'snr': 'category'})
    
    g = sns.scatterplot(x='di', y='diff', hue='snr', data=_df, ax=a)
    #a.scatter(b, diff, s=40, edgecolor='k')
    a.set_xlabel('Behavior performance (DI)')
    a.set_ylabel(r"$\Delta r_{sc}$")
    a.axhline(0, linestyle='--', color='k')
    r, p = ss.pearsonr(b, diff)
    a.set_title(f"Time bin: {tb}\nr: {round(r, 3)}, pval:  {round(p, 3)}")
    if i == 0:
        g.legend(frameon=False)
    else:
        g.legend([])
f.tight_layout()

# same as above for the TIN / BVT data
di = 'DIref'
mask = (df.batch==302) | (df.batch==324)
tbin_302_324 = ['0_0.1', '0.1_0.2', '0.2_0.3', '0.3_0.4', '0.4_0.5']
f, ax = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
for i, (a, tb) in enumerate(zip(ax.flatten(), tbin_302_324)):
    diff = df[mask & (df.tbin==tb) & ((df.pa<0.05) | (df.pp<0.05)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()['diff']
    b = df[mask & (df.tbin==tb) & ((df.pa<0.05) | (df.pp<0.05)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()[di]
    snr = b.index.get_level_values(0)
    b = b[~diff.isna()].values
    snr = snr[~diff.isna()]
    diff = diff[~diff.isna()].values
    _df = pd.DataFrame(columns=['diff', 'di', 'snr'], data=np.stack([diff, b, snr]).T)
    _df = _df.astype({'snr': 'category'})
    
    g = sns.scatterplot(x='di', y='diff', hue='snr', data=_df, ax=a)
    #a.scatter(b, diff, s=40, edgecolor='k')
    a.set_xlabel('Behavior performance (DI)')
    a.set_ylabel(r"$\Delta r_{sc}$")
    a.axhline(0, linestyle='--', color='k')
    r, p = ss.pearsonr(b, diff)
    a.set_title(f"Time bin: {tb}\nr: {round(r, 3)}, pval:  {round(p, 3)}")
    if i == 0:
        g.legend(frameon=False)
    else:
        g.legend([])
f.tight_layout()

# combine the two A1 datasets
mask1 = df.batch == 307
mask2 = (df.batch==302) | (df.batch==324)
alpha = 0.05
title = ['-0.1 : 0 sec', '0 : 0.1 sec', '0.1 : 0.2 sec', '0.2 : 0.3 sec', '0.4 : 0.5 sec']
t307 = tbin307[:5]
tOther = tbin_302_324
f, ax = plt.subplots(1, 5, figsize=(12, 3), sharey=True)
for a, tit, tb3, tbo in zip(ax.flatten(), title, t307, tOther):
    #diff = df[mask1 & (df.tbin==tb3) & ((df.pa<alpha) | (df.pp<alpha))].groupby(by='site').mean()['diff']
    #b = df[mask1 & (df.tbin==tb3) & ((df.pa<alpha) | (df.pp<alpha))].groupby(by='site').mean()[di]
    diff = df[mask1 & (df.tbin==tb3) & ((df.pa<alpha) | (df.pp<alpha)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()['diff']
    b = df[mask1 & (df.tbin==tb3) & ((df.pa<alpha) | (df.pp<alpha)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()[di]
    diff = pd.concat([diff,
        df[mask2 & (df.tbin==tbo) & ((df.pa<alpha) | (df.pp<alpha)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()['diff']])
    b = pd.concat([b, 
        df[mask2 & (df.tbin==tbo) & ((df.pa<alpha) | (df.pp<alpha)) & (df.snr!=-np.inf)].groupby(by=['snr', 'site']).mean()[di]])
    b = b[~diff.isna()]
    diff = diff[~diff.isna()]
    a.scatter(b, diff, s=20, edgecolor='k')
    a.set_xlabel('Behavior performance (DI)')
    a.set_ylabel(r"$\Delta r_{sc}$")
    a.axhline(0, linestyle='--', color='k')
    a.axvline(0.5, linestyle='--', color='k')
    r, p = ss.pearsonr(b, diff)
    a.set_title(f"Time bin: {tit}\nr: {round(r, 3)}, pval:  {round(p, 3)}")

f.tight_layout()

plt.show()