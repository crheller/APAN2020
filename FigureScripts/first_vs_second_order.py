"""
Show changes in dU vs. changes in noise variance.
    * Point is that both are striking, only one (?) correlated with behavior
    * also show pupil / behavior regression?
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
    
df = pd.read_pickle(DIR+"results/res_pr.pickle")
df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
di_metric = 'DI'  # for this data, DI = DIref if df.aref_tar = True
dp_metric = 'dp_opt_sqrt'
diff_norm = False

mask = ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.cat_tar & df.batch.isin([302, 307, 324])
df = df[mask]

df['lambda_tot'] = df['evals'].apply(lambda x: sum(x))
df['dU_mag'] = df['dU'].apply(lambda x: np.linalg.norm(x))


f, ax = plt.subplots(1, 2, figsize=(8, 4))

# first order changes
ax[0].scatter(df[~df.active]['dU_mag'], df[df.active]['dU_mag'], s=30, edgecolor='white', color='k')
m = np.max(ax[0].get_xlim()+ax[1].get_ylim())
mi = np.min(ax[0].get_xlim()+ax[1].get_ylim())
ax[0].plot([mi, m], [mi, m], '--', color='grey')
ax[0].set_xlabel("Passive")
ax[0].set_ylabel("Active")
ax[0].set_title("Signal magnitude")

# second order changes
ax[1].scatter(df[~df.active]['lambda_tot'], df[df.active]['lambda_tot'], s=30, edgecolor='white', color='k')
m = np.max(ax[1].get_xlim()+ax[1].get_ylim())
mi = np.min(ax[1].get_xlim()+ax[1].get_ylim())
ax[1].plot([mi, m], [mi, m], '--', color='grey')
ax[1].set_xlabel("Passive")
ax[1].set_ylabel("Active")
ax[1].set_title("Noise Variance")

f.tight_layout()

plt.show()


# change in noise correlated with behavior, change in dU not. Both contribute to "optimal" dprime (which itself is uncorrelated with behavior)
diff = (df[~df.active]['lambda_tot']-df[df.active]['lambda_tot']) / (df[~df.active]['lambda_tot']+df[df.active]['lambda_tot'])
ss.pearsonr(diff, df[~df.active]['DI']) 