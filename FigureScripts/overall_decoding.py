"""
3 x 3 set of results:
raw, pupil removed, pupil + behavior removed for:
    1) tar vs. tar decoding
    2) ref vs. ref decoding
    3) cat vs. tar decoding
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

dp_metric = 'dp_opt_sqrt'
s = 50
cmap = {
    'tar_tar': 'coral',
    'ref_ref': 'forestgreen',
    'tar_cat': 'black'
}
    
df = pd.read_pickle(DIR+"results/res.pickle")
df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
dfp = pd.read_pickle(DIR+"results/res_pr.pickle")
dfp['dp_opt_sqrt'] = np.sqrt(dfp['dp_opt'])
dfp['dp_diag_sqrt'] = np.sqrt(dfp['dp_diag'])
dfbp = pd.read_pickle(DIR+"results/res_pr_br.pickle")
dfbp['dp_opt_sqrt'] = np.sqrt(dfbp['dp_opt'])
dfbp['dp_diag_sqrt'] = np.sqrt(dfbp['dp_diag'])

mask1 = df.tdr_overall & ~df.pca & ~df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & df.aref_tar
mask2 = df.tdr_overall & ~df.pca & ~df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & df.tar_tar
mask3 = df.tdr_overall & ~df.pca & ~df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & df.ref_ref

f, ax = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# RAW DATA

ax[0, 0].scatter(df[mask1 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 df[mask1 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['tar_cat'])
ax[0, 0].set_title('Target vs. Catch')

ax[1, 0].scatter(df[mask2 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 df[mask2 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['tar_tar'])
ax[1, 0].set_title('Target vs. Target')

ax[2, 0].scatter(df[mask3 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 df[mask3 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['ref_ref'])
ax[2, 0].set_title('Reference vs. Reference')

# LINEAR PUPIL CORRECTION

ax[0, 1].scatter(dfp[mask1 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 dfp[mask1 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['tar_cat'])
ax[0, 1].set_title('Target vs. Catch')

ax[1, 1].scatter(dfp[mask2 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 dfp[mask2 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['tar_tar'])
ax[1, 1].set_title('Target vs. Target')

ax[2, 1].scatter(dfp[mask3 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 dfp[mask3 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['ref_ref'])
ax[2, 1].set_title('Reference vs. Reference')

# LINEAR PUPIL + BEHAVIOR CORRECTION

ax[0, 2].scatter(dfbp[mask1 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 dfbp[mask1 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['tar_cat'])
ax[0, 2].set_title('Target vs. Catch')

ax[1, 2].scatter(dfbp[mask2 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 dfbp[mask2 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['tar_tar'])
ax[1, 2].set_title('Target vs. Target')

ax[2, 2].scatter(dfbp[mask3 & ~df.active].groupby(by='site').mean()[dp_metric], 
                 dfbp[mask3 & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['ref_ref'])
ax[2, 2].set_title('Reference vs. Reference')

mi = np.min(ax[0, 0].get_xlim()+ax[0, 0].get_ylim())
ma = np.max(ax[0, 0].get_xlim()+ax[0, 0].get_ylim())
for a in ax.flatten():
    a.plot([mi, ma], [mi, ma], '--', color='grey')
    a.set_xlabel('Passive')
    a.set_ylabel('Active')

f.tight_layout()

f.savefig(DIR + 'pyfigures/overall_decoding_scatter.svg')

# summary barplots for each row (each correction condition)
f, ax = plt.subplots(3, 1, figsize=(3, 9), sharey=True, sharex=True)

ax[0].bar([0, 1, 2],
            [(df[mask1 & df.active].groupby(by='site').mean()[dp_metric] - df[mask1 & ~df.active].groupby(by='site').mean()[dp_metric]).mean(),
            (dfp[mask1 & df.active].groupby(by='site').mean()[dp_metric] - dfp[mask1 & ~df.active].groupby(by='site').mean()[dp_metric]).mean(),
            (dfbp[mask1 & df.active].groupby(by='site').mean()[dp_metric] - dfbp[mask1 & ~df.active].groupby(by='site').mean()[dp_metric]).mean()],
            yerr=[(df[mask1 & df.active].groupby(by='site').mean()[dp_metric] - df[mask1 & ~df.active].groupby(by='site').mean()[dp_metric]).sem(),
                (dfp[mask1 & df.active].groupby(by='site').mean()[dp_metric] - dfp[mask1 & ~df.active].groupby(by='site').mean()[dp_metric]).sem(),
                (dfbp[mask1 & df.active].groupby(by='site').mean()[dp_metric] - dfbp[mask1 & ~df.active].groupby(by='site').mean()[dp_metric]).sem()],
            edgecolor=cmap['tar_cat'], color='none',
            lw=3, error_kw=dict(lw=3, capsize=3, capthick=2)
            )
ax[0].set_ylabel(r"$\Delta d'$")
ax[0].set_xlim((-2, 4))
ax[0].set_title('Target vs. Catch')

# pupil correction
ax[1].bar([0, 1, 2],
            [(df[mask2 & df.active].groupby(by='site').mean()[dp_metric] - df[mask2 & ~df.active].groupby(by='site').mean()[dp_metric]).mean(),
            (dfp[mask2 & df.active].groupby(by='site').mean()[dp_metric] - dfp[mask2 & ~df.active].groupby(by='site').mean()[dp_metric]).mean(),
            (dfbp[mask2 & df.active].groupby(by='site').mean()[dp_metric] - dfbp[mask2 & ~df.active].groupby(by='site').mean()[dp_metric]).mean()],
            yerr=[(df[mask2 & df.active].groupby(by='site').mean()[dp_metric] - df[mask2 & ~df.active].groupby(by='site').mean()[dp_metric]).sem(),
                (dfp[mask2 & df.active].groupby(by='site').mean()[dp_metric] - dfp[mask2 & ~df.active].groupby(by='site').mean()[dp_metric]).sem(),
                (dfbp[mask2 & df.active].groupby(by='site').mean()[dp_metric] - dfbp[mask2 & ~df.active].groupby(by='site').mean()[dp_metric]).sem()],
            edgecolor=cmap['tar_tar'], color='none',
            lw=3, error_kw=dict(lw=3, capsize=3, capthick=2)
            )
ax[1].set_ylabel(r"$\Delta d'$")
ax[1].set_xlim((-2, 4))
ax[1].set_title('Target vs. Target')

# pupil + behavior correction
ax[2].bar([0, 1, 2],
            [(df[mask3 & df.active].groupby(by='site').mean()[dp_metric] - df[mask3 & ~df.active].groupby(by='site').mean()[dp_metric]).mean(),
            (dfp[mask3 & df.active].groupby(by='site').mean()[dp_metric] - dfp[mask3 & ~df.active].groupby(by='site').mean()[dp_metric]).mean(),
            (dfbp[mask3 & df.active].groupby(by='site').mean()[dp_metric] - dfbp[mask3 & ~df.active].groupby(by='site').mean()[dp_metric]).mean()],
            yerr=[(df[mask3 & df.active].groupby(by='site').mean()[dp_metric] - df[mask3 & ~df.active].groupby(by='site').mean()[dp_metric]).sem(),
                (dfp[mask3 & df.active].groupby(by='site').mean()[dp_metric] - dfp[mask3 & ~df.active].groupby(by='site').mean()[dp_metric]).sem(),
                (dfbp[mask3 & df.active].groupby(by='site').mean()[dp_metric] - dfbp[mask3 & ~df.active].groupby(by='site').mean()[dp_metric]).sem()],
            edgecolor=cmap['ref_ref'], color='none',
            lw=3, error_kw=dict(lw=3, capsize=3, capthick=2)
            )
ax[2].set_ylabel(r"$\Delta d'$")
ax[2].set_xlim((-2, 4))
ax[2].set_title('Reference vs. Reference')

f.tight_layout()

f.savefig(DIR + 'pyfigures/overall_decoding_bar.svg')

plt.show()