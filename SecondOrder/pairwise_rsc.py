"""
Exploratory analysis of pairwise rsc
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle('/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/rsc_df.pickle')
tbin = ['0_0.1', '0.1_0.2', '0.2_0.3', '0.3_0.4', '0.4_0.5']
di = 'DIrefall'
# plot active/passive as fn of time bin
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

x = np.arange(len(df.tbin.unique()))  # the label locations
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

# change in noise corr. vs DI grouped by area (subplot) and snr (color) and timebin (color)
f, ax = plt.subplots(2, len(tbin), figsize=(15, 6), sharex=True, sharey=True)

for i, tb in enumerate(tbin):
    # A1
    a1 = df[(df.tbin==tb) & (df.area=='A1')].groupby(by=['site', 'snr']).mean()[['active', 'passive', di]]
    a1 = a1[a1[di]!=np.inf]
    a1[r"$\Delta r_{sc}$"] = a1['passive'] - a1['active']
    a1['snr'] = a1.index.get_level_values('snr')
    a1.at[a1.snr==np.inf, 'snr'] = 5
    a1['site'] = a1.index.get_level_values('site')

    g = sns.scatterplot(x=di, y=r"$\Delta r_{sc}$", hue='snr', data=a1, ax=ax[0, i], s=40, edgecolor='k')
    if i != 0:
        g.legend([])
    else:
        g.legend(frameon=False)
    ax[0, i].axvline(0.5, linestyle='--', color='k')
    ax[0, i].axhline(0, linestyle='--', color='k')


    # PEG
    peg = df[(df.tbin==tb) & (df.area=='PEG')].groupby(by=['site', 'snr']).mean()[['active', 'passive', di]]
    peg = peg[peg[di]!=np.inf]
    peg[r"$\Delta r_{sc}$"] = peg['passive'] - peg['active']
    peg['snr'] = peg.index.get_level_values('snr')
    peg.at[peg.snr==np.inf, 'snr'] = 5
    peg['site'] = peg.index.get_level_values('site')

    g = sns.scatterplot(x=di, y=r"$\Delta r_{sc}$", hue='snr', data=peg, ax=ax[1, i], s=40, edgecolor='k')
    if i != 0:
        g.legend([])
    else:
        g.legend(frameon=False)
    ax[1, i].axvline(0.5, linestyle='--', color='k')
    ax[1, i].axhline(0, linestyle='--', color='k')


f.tight_layout()

plt.show()
    