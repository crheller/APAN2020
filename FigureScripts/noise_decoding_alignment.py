"""
Illustrate alignment of decoding axis (and/or dU) with noise axis.
Show as function of target SNR.
"""
from settings import DIR
from scipy.optimize import curve_fit
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle(DIR + 'results/res.pickle')
df['fdiff'] = round(abs(np.log2(df['f1'] / df['f2'])), 1)
lv = pickle.load(open(DIR + '/results/drsc_axes.pickle', "rb"))
# remember, in the lv dict for batch 307, "catch" = REFERENCE.

# add column(s) for lv projection into decoding space / alignment with relevant axes for each pair
projection = []
mag = []
cosdU = []
for r in df.iterrows():
    lvax = lv[r[1]['site']]['tarOnly']['evecs'][:, 0]
    proj = lvax.dot(r[1]['dr_weights'].T)
    m = np.linalg.norm(proj)
    #proj_norm = proj / np.linalg.norm(m)
    # actually, don't normalize. Want "raw" alignment of the axis
    cos = abs((r[1]['dU'] / np.linalg.norm(r[1]['dU'])).dot(proj)[0])

    projection.append(proj)
    mag.append(m)
    cosdU.append(cos)

df['lv'] = projection
df['lv_mag'] = mag
df['lv_cos_dU'] = cosdU

# Look at alignment of noise corr with dU as fn of SNR
f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# batch 324 only (where we have a "real" catch along SNR axis)
mask = df.cat_tar & (df.batch.isin([324])) & ~df.tdr_overall & ~df.active & ~df.pca & df.tdr_fixedNoise
grp = df[mask].groupby(by=['snr1', 'site']).mean()
snrsAll = grp.index.get_level_values('snr1').unique()
for s in grp.index.get_level_values('site').unique():
    snrs = grp.loc[pd.IndexSlice[:, s], :].index.get_level_values('snr1')
    snridx = [True if s in snrs else False for s in snrsAll]
    x = np.arange(0, len(snrsAll))[snridx]
    cos = grp.loc[pd.IndexSlice[:, s], :]['lv_cos_dU']
    ax[0].plot(x, cos, color='lightgrey')
ax[0].plot(np.arange(0, len(snrsAll)), df[mask].groupby(by=['snr1']).mean()['lv_cos_dU'], 'o-', lw=2,
            color='k')
ax[0].set_xticks(np.arange(0, len(snrsAll)))
ax[0].set_xticklabels([str(x) for x in snrsAll])
ax[0].set_xlabel('Target SNR')
ax[0].set_ylabel('Noise vs. Signal alignment')
ax[0].set_title('Narrowband noise')

# batch 307 (broadband noise)
mask = df.aref_tar & (df.batch==307) & ~df.tdr_overall & ~df.active & ~df.pca & df.tdr_fixedNoise
grp = df[mask].groupby(by=['snr2', 'site']).mean()
snrsAll = grp.index.get_level_values('snr2').unique()
for s in grp.index.get_level_values('site').unique():
    snrs = grp.loc[pd.IndexSlice[:, s], :].index.get_level_values('snr2')
    snridx = [True if s in snrs else False for s in snrsAll]
    x = np.arange(0, len(snrsAll))[snridx]
    cos = grp.loc[pd.IndexSlice[:, s], :]['lv_cos_dU']
    ax[1].plot(x, cos, color='grey')

ax[1].set_xticks(np.arange(0, len(snrsAll)))
ax[1].set_xticklabels([str(x) for x in snrsAll])
ax[1].set_xlabel('Target SNR')
ax[1].set_ylabel('Noise vs. Signal alignment')
ax[1].set_title('Broadband noise')

f.tight_layout()


# break down the 324 / 325? data more: by frequency dim AND SNR dim.

mask = df.ref_tar & df.batch.isin([324]) & ~df.tdr_overall & ~df.active & ~df.pca & df.tdr_fixedNoise
f, ax = plt.subplots(1, 1, figsize=(5, 5))
for snr in df[mask].snr2.unique():
    grp = df[mask & (df.snr2==snr)].groupby(by='fdiff').mean()
    fdiff = grp.index.get_level_values('fdiff')
    cos = grp['lv_cos_dU']
    pfit = np.poly1d(np.polyfit(fdiff, cos, 2))
    x = np.arange(0, max(fdiff), 0.01)
    ax.plot(x, pfit(x))
    ax.scatter(fdiff, cos, s=30, edgecolor='white', color=ax.get_lines()[-1].get_color(), label=snr)

ax.legend(frameon=False)
ax.set_ylabel('Noise vs. Signal alignment')
ax.set_xlabel('Octave separation')

f.tight_layout()




# 324/325 decoding changes vs. SNR and decoding changes vs. noise alignment
amask = df.aref_tar & (df.batch.isin([302, 324, 325])) & ~df.tdr_overall & ~df.pca & df.active & df.tdr_fixedNoise
pmask = df.aref_tar & (df.batch.isin([302, 324, 325])) & ~df.tdr_overall & ~df.pca & ~df.active & df.tdr_fixedNoise
grpa = df[amask].groupby(by=['snr2', 'site']).mean()
grpp = df[pmask].groupby(by=['snr2', 'site']).mean()
dp_metric = 'dp_opt'

f, ax = plt.subplots(1, 1, figsize=(5, 5))

for snr in grpa.index.get_level_values('snr2').unique():
    a = grpa.loc[pd.IndexSlice[snr, :], :]
    p = grpp.loc[pd.IndexSlice[snr, :], :]
    ddp = (a[dp_metric] - p[dp_metric]) / (a[dp_metric] + p[dp_metric])
    nvs = grpp.loc[pd.IndexSlice[snr, :], :]['lv_cos_dU']

    ax.scatter(nvs, ddp, s=50, edgecolor='white', label=f"{snr} dB SNR")

ax.legend()
ax.set_xlabel('Noise vs. Signal Alignment')
ax.set_ylabel(r"$\Delta d'^2$")

f.tight_layout()

plt.show()