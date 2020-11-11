"""
Exploratory plots of dprime vs. n Noise dims 
Idea is to replicate something like the Ni et al. paper
"""
from settings import DIR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

choice = pd.read_pickle(DIR + 'results/res_choice_decoder.pickle')
stimulus = pd.read_pickle(DIR + 'results/res_stimulus_decoder.pickle')


f, ax = plt.subplots(2, 2, figsize=(8, 8))

d = stimulus 
# first plot for overall noise (pca axes)
m = d[d['axes']=='pca'].groupby(by=['snr', 'nDim']).mean()
sem = d[d['axes']=='pca'].groupby(by=['snr', 'nDim']).sem()
ma = m.groupby(level=0).max()
m['snr'] = m.index.get_level_values(level=0)
nDim = m.index.get_level_values(level=1).unique()
ax[0, 0].plot(nDim, m[m.snr==-5]['dprime'] / ma.iloc[0][0], lw=2, label='-5 dB')
ax[0, 0].fill_between(nDim, (m[m.snr==-5]['dprime'].values / ma.iloc[0][0])-(sem[m.snr==-5]['dprime'].values / ma.iloc[0][0]), 
                    (m[m.snr==-5]['dprime'].values / ma.iloc[0][0]) + (sem[m.snr==-5]['dprime'].values / ma.iloc[0][0]), alpha=0.3, lw=0)
ax[0, 0].plot(nDim, m[m.snr==0]['dprime'] / ma.iloc[1][0], lw=2, label='0 dB')
ax[0, 0].fill_between(nDim, (m[m.snr==0]['dprime'].values / ma.iloc[1][0]) - (sem[m.snr==0]['dprime'].values / ma.iloc[1][0]), 
                    (m[m.snr==0]['dprime'].values / ma.iloc[1][0]) + (sem[m.snr==0]['dprime'].values / ma.iloc[1][0]), alpha=0.3, lw=0)
ax[0, 0].plot(nDim, m[m.snr==np.inf]['dprime'] / ma.iloc[2][0], lw=2, label='Inf dB')
ax[0, 0].fill_between(nDim, (m[m.snr==np.inf]['dprime'].values / ma.iloc[2][0]) - (sem[m.snr==np.inf]['dprime'].values / ma.iloc[2][0]), 
                    (m[m.snr==np.inf]['dprime'].values / ma.iloc[2][0]) + (sem[m.snr==np.inf]['dprime'].values / ma.iloc[2][0]), alpha=0.3, lw=0)
ax[0, 0].set_title('Noise dims, stim decoding')

# second plot for delta noise (delta axes)
m = d[d['axes']=='delta'].groupby(by=['snr', 'nDim']).mean()
sem = d[d['axes']=='delta'].groupby(by=['snr', 'nDim']).sem()
sem /= m.max()
m /= m.max()
m['snr'] = m.index.get_level_values(level=0)
nDim = m.index.get_level_values(level=1).unique()
ax[0, 1].plot(nDim, m[m.snr==-5]['dprime'], lw=2, label='-5 dB')
ax[0, 1].fill_between(nDim, m[m.snr==-5]['dprime'].values-sem[m.snr==-5]['dprime'].values, 
                    m[m.snr==-5]['dprime'].values+sem[m.snr==-5]['dprime'].values, alpha=0.3, lw=0)
ax[0, 1].plot(nDim, m[m.snr==0]['dprime'], lw=2, label='0 dB')
ax[0, 1].fill_between(nDim, m[m.snr==0]['dprime'].values-sem[m.snr==0]['dprime'].values, 
                    m[m.snr==0]['dprime'].values+sem[m.snr==0]['dprime'].values, alpha=0.3, lw=0)
ax[0, 1].plot(nDim, m[m.snr==np.inf]['dprime'], lw=2, label='Inf dB')
ax[0, 1].fill_between(nDim, m[m.snr==np.inf]['dprime'].values-sem[m.snr==np.inf]['dprime'].values, 
                    m[m.snr==np.inf]['dprime'].values+sem[m.snr==np.inf]['dprime'].values, alpha=0.3, lw=0)
ax[0, 1].set_title('Delta noise dims, stim decoding')
ax[0, 1].legend(frameon=False)

d = choice
# first plot for overall noise (pca axes)
m = d[(d['axes']=='pca') & (d['soundCategory']=='target')].groupby(by=['nDim']).mean()['dprime']
sem = d[(d['axes']=='pca') & (d['soundCategory']=='target')].groupby(by=['nDim']).sem()['dprime']
sem /= m.max()
m /= m.max()
ax[1, 0].plot(nDim, m, lw=2)
ax[1, 0].fill_between(nDim, m-sem, m+sem, lw=0, alpha=0.3)
ax[1, 0].set_title('Noise dims, choice decoding')
ax[1, 0].axhline(0, linestyle='--', color='grey', lw=2)

# second plot for delta noise (delta axes)
m = d[(d['axes']=='delta') & (d['soundCategory']=='target')].groupby(by=['nDim']).mean()['dprime']
sem = d[(d['axes']=='delta') & (d['soundCategory']=='target')].groupby(by=['nDim']).sem()['dprime']
sem /= m.max()
m /= m.max()
ax[1, 1].plot(nDim, m, lw=2)
ax[1, 1].fill_between(nDim, m-sem, m+sem, lw=0, alpha=0.3)
ax[1, 1].set_title('Delta noise dims, choice decoding')
ax[1, 1].axhline(0, linestyle='--', color='grey', lw=2)

ylim = (np.min(ax[1, 0].get_ylim() + ax[1, 1].get_ylim()), np.max(ax[1, 0].get_ylim() + ax[1, 1].get_ylim()))
ax[1, 0].set_ylim(ylim)
ax[1, 1].set_ylim(ylim)

f.tight_layout()

plt.show()