"""
Regress out first order effects, look at residual change in decoding for each target SNR

Hypothesis: The delta dprime for low SNR targets will be least affected by the regression.
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

df = pd.read_pickle(DIR+"results/res.pickle")
dfr = pd.read_pickle(DIR+"results/res_deflate.pickle")
df = df.astype({'snr1': 'category'})
dp_metric = 'dp_opt'
mask = df.tdr_overall & df.cat_tar & ~df.pca & df.batch.isin([324, 325])

# just scatter plot of regression results vs. raw results
f, ax = plt.subplots(1, 1, figsize=(5, 5))

sns.scatterplot((df[mask & df.active][dp_metric] - df[mask & ~df.active][dp_metric]), #/ (df[mask & df.active][dp_metric] + df[mask & ~df.active][dp_metric]),
            (dfr[mask & df.active][dp_metric] - dfr[mask & ~df.active][dp_metric]), #/ (dfr[mask & df.active][dp_metric] + dfr[mask & ~df.active][dp_metric]),
            s=50, edgecolor='k', ax=ax, hue=df[mask & df.active]['snr1'])
ax.plot([-0.5, 35], [-0.5, 35], 'k--')
ax.axhline(0, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.set_title(r"$\Delta d'^2$")
ax.set_xlabel(r"Raw data")
ax.set_ylabel(r"First-order corrected")

f.tight_layout()

# fraction of effects explained by first order
snrs = df[mask & df.active].snr1.unique()
f, ax = plt.subplots(1, len(snrs), figsize=(12, 4), sharex=True, sharey=True)

frac_m = []
frac_sem = []
for a, snr in zip(ax.flatten(), snrs):
    num = (dfr[mask & df.active & (df.snr1==snr)][dp_metric] - dfr[mask & ~df.active & (df.snr1==snr)][dp_metric]) / \
            (dfr[mask & df.active & (df.snr1==snr)][dp_metric] + dfr[mask & ~df.active & (df.snr1==snr)][dp_metric])
    den = (df[mask & df.active & (df.snr1==snr)][dp_metric] - df[mask & ~df.active & (df.snr1==snr)][dp_metric]) / \
            (df[mask & df.active & (df.snr1==snr)][dp_metric] + df[mask & ~df.active & (df.snr1==snr)][dp_metric]) 
    frac = num / den
    frac_m.append(frac.mean())
    frac_sem.append(frac.sem())

    a.scatter(num, den, s=50, edgecolor='white')
    a.set_title(r"$\Delta d'^2$"+f"\nSNR: {snr} dB")
    a.set_xlabel('Removed noise correlation')
    a.set_ylabel('Raw Data')
    a.plot([-1, 1], [-1, 1], 'k--')
    a.axhline(0, linestyle='--', color='k')
    a.axvline(0, linestyle='--', color='k')
#ax.bar(range(len(snrs)), frac_m, yerr=frac_sem, edgecolor='k', lw=2)
#ax.set_xticks(range(len(snrs)))
#ax.set_xticklabels(snrs)
#ax.set_xlabel('SNR')
#ax.set_ylabel(r"Fraction $\Delta d'$ expained"+"\nby first-order")

f.tight_layout()

plt.show()