"""
Load "noise correlation latent variable". See how it projects into the decoding space
    (what's it's magnitude? How well does it align with coding dims for different stimuli? In different areas?)
"""
from settings import DIR
import pickle
import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_pickle(DIR + 'results/res.pickle')
dfr = pd.read_pickle(DIR + 'results/res_deflate.pickle')
lv = pickle.load(open(DIR + 'results/drsc_axes.pickle', "rb"))

mask = df.cat_tar & ~df.tdr_overall & ~df.pca & df.batch.isin([324, 325]) #& (df.f1==df.f2)
df = df[mask]
dfr = dfr[mask]
# add column for lv projection into decoding space for each pair
projection = []
mag = []
cosdU = []
cosWopt = []
for r in df.iterrows():
    lvax = lv[r[1]['site']]['tarCat']['evecs'][:, 0]
    proj = lvax.dot(r[1]['dr_weights'].T)
    m = np.linalg.norm(proj)
    proj_norm = proj / np.linalg.norm(m)
    cos = abs((r[1]['dU'] / np.linalg.norm(r[1]['dU'])).dot(proj)[0])
    cosw = abs((r[1]['wopt'] / np.linalg.norm(r[1]['wopt'])).T.dot(proj)[0])
    projection.append(proj)
    mag.append(m)
    cosdU.append(cos)
    cosWopt.append(cosw)

df['lv'] = projection
df['lv_mag'] = mag
df['lv_cos_dU'] = cosdU
df['lv_cos_wopt'] = cosWopt

val = 'dp_opt'
f, ax = plt.subplots(1, 1)

x = df[df.active]['lv_cos_dU']
y = (df[df.active][val] - df[~df.active][val]) / (df[df.active][val] + df[~df.active][val])
y2 = (dfr[df.active][val] - dfr[~df.active][val]) / (dfr[df.active][val] + dfr[~df.active][val])
ax.scatter(x, y, s=50, edgecolor='k', label='raw')
ax.scatter(x, y2, s=50, edgecolor='k', label='deflated')
ax.scatter(x, y-y2, s=50, edgecolor='k', label='diff')
ax.axhline(0, linestyle='--', color='k')
ax.legend()
r, p = ss.pearsonr(x, y)
ax.set_title(f"r: {round(r, 3)} p: {round(p, 3)}")
# alignment of noise corr axis as fn of snr
df[df.active].groupby(by='snr1').mean()['lv_cos_dU']


plt.show()


