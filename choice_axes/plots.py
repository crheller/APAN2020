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
import pandas as pd
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

choice = pd.read_pickle(DIR + 'results/res_choice_decoder.pickle')
stimulus = pd.read_pickle(DIR + 'results/res_stimulus_decoder.pickle')


# Ni et al - like figure
for axes in ['target', 'catch', 'pca', 'delta', 'tarCat']:
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    snrs = [0]

    # RAW DPRIMES
    # stimulus decoding
    data = stimulus[(stimulus['axes']==axes) & stimulus.snr.isin(snrs)]
    normdata = np.zeros((len(data.nDim.unique()), len(data.site.unique())))
    for i, s in enumerate(data.site.unique()):
        v = data[data.site==s].groupby(by='nDim').mean()['dprime']
        normdata[:, i] = v
    m = normdata.mean(axis=-1)
    sem = normdata.std(axis=-1) / np.sqrt(normdata.shape[-1])
    x = data.nDim.unique()
    ax.errorbar(x=x, y=m, yerr=sem,
                    capsize=3, lw=1, label='stimulus')

    # choice decoding
    data = choice[(choice['axes']==axes) & (choice.soundCategory=='target')]
    normdata = np.zeros((len(data.nDim.unique()), len(data.site.unique())))
    for i, s in enumerate(data.site.unique()):
        v = data[data.site==s].groupby(by='nDim').mean()['dprime']
        normdata[:, i] = v
    m = normdata.mean(axis=-1)
    sem = normdata.std(axis=-1) / np.sqrt(normdata.shape[-1])
    x = data.nDim.unique()
    ax.errorbar(x=x, y=m, yerr=sem,
                    capsize=3, lw=1, label='target choice')

    data = choice[(choice['axes']==axes) & (choice.soundCategory=='catch')]
    normdata = np.zeros((len(data.nDim.unique()), len(data.site.unique())))
    for i, s in enumerate(data.site.unique()):
        v = data[data.site==s].groupby(by='nDim').mean()['dprime']
        normdata[:, i] = v
    m = np.nanmean(normdata, axis=-1)
    sem = np.nanstd(normdata, axis=-1) / np.sqrt(normdata.shape[-1])
    x = data.nDim.unique()
    ax.errorbar(x=x, y=m, yerr=sem,
                    capsize=3, lw=1, label='catch choice')


    ax.axhline(0, linestyle='--', color='grey')
    ax.legend(frameon=False)
    ax.set_xlabel('nPCs')
    ax.set_ylabel(r"$d'$")

    ax.set_title(axes)

    f.tight_layout()

plt.show()