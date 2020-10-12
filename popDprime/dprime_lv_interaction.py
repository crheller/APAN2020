"""
Load "noise correlation latent variable". See how it projects into the decoding space
    (what's it's magnitude? How well does it align with coding dims for different stimuli? In different areas?)
"""

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle('/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/res_pr.pickle')
lv = pickle.load(open('/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/drsc_axes.pickle', "rb"))

mask = df.cat_tar & ~df.tdr_overall & ~df.pca & (df.f1==df.f2)
df = df[mask]

# add column for lv projection into decoding space for each pair
projection = []
mag = []
cosdU = []
for r in df.iterrows():
    lvax = lv[r[1]['site']]['tarCat']['evecs'][:, 0]
    proj = lvax.dot(r[1]['dr_weights'].T)
    m = np.linalg.norm(proj)
    proj_norm = proj / np.linalg.norm(m)
    cos = abs((r[1]['dU'] / np.linalg.norm(r[1]['dU'])).dot(proj_norm)[0])

    projection.append(proj)
    mag.append(m)
    cosdU.append(cos)

df['lv'] = projection
df['lv_mag'] = mag
df['lv_cos_dU'] = cosdU

val = 'dp_opt'
f, ax = plt.subplots(1, 1)

ax.scatter(df[~df.active]['lv_cos_dU'] * df[~df.active]['lv_mag'], df[df.active][val] - df[~df.active][val], s=50, edgecolor='k')


# alignment of noise corr axis as fn of snr
df[df.active].groupby(by='snr1').mean()['lv_cos_dU']


plt.show()


'''
Notes 10.11.2020

Noise suppression seems low dim (delta rsc plots, compare with raw pairwise noise correlations)

Noise is not aligned with discrimination axis for simple task (pure tone), but is more aligned for low SNR targets

Noise changes (maybe) correlated with behavior, discrim not (at least for PTD data... need to link this with TBP)


'''


