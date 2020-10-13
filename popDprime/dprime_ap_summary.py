"""
Overall look at behavior-dependent change in decoding per site
"""
import scipy.stats as ss
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle('/auto/users/hellerc/code/projects/APAN2020/results/res.pickle')
df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
# for 302, catch = n.r. tone
# for 307, catch = REFERENCE (broad band noise)
# for 324/325, catch = noise alone


# very crude look at data. One point for each target/catch combo across all sites
mask = df.tdr_overall & ((df.cat_tar & (df.batch==324) & (df.f1==df.f2)) | (df.aref_tar & (df.batch==307)) | (df.aref_tar & (df.batch==302))) & ~df.pca

dpval = 'dp_opt_sqrt'
f, ax = plt.subplots(1, 1, figsize=(5, 5))

for bat in df.batch.unique():
    ax.scatter(df[mask & (df.batch==bat) & ~df.active][dpval], 
               df[mask & (df.batch==bat) & df.active][dpval], 
                        s=30, edgecolor='k', label=int(bat))

ax.plot([df[mask][dpval].min(), df[mask][dpval].max()], 
            [df[mask][dpval].min(), df[mask][dpval].max()], 'k--')
ax.legend(frameon=False)
ax.set_xlabel('Passive')
ax.set_ylabel('Active')
ax.set_title(r"$d'^2$")

f.tight_layout()

# compare behavioral performance with change in DI
dpval = 'dp_diag'
mask = df.aref_tar & ~df.tdr_overall
f, ax = plt.subplots(1, 1, figsize=(5, 5))

for bat in [307]: #df.batch.unique():
    diff = (df[mask & (df.batch==bat) & df.active][dpval] - df[mask & (df.batch==bat) & ~df.active][dpval]) / \
                        (df[mask & (df.batch==bat) & df.active][dpval] + df[mask & (df.batch==bat) & ~df.active][dpval])
    di = df[mask & (df.batch==bat) & df.active]['DI']
    r, p = ss.pearsonr(di, diff)
    ax.scatter(di, diff, s=30, edgecolor='white', label=f"{int(bat)}: r: {round(r, 3)}, pval: {round(p, 3)}")


ax.axhline(0, linestyle='--', color='k')
ax.axvline(0.5, linestyle='--', color='k')
ax.legend(frameon=False)
ax.set_xlabel('DI')
ax.set_ylabel(r"$\Delta d'$")
ax.set_title(r"$d'^2$")

f.tight_layout()


plt.show()
