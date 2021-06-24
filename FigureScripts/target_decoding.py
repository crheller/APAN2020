"""
Target vs. target discrimination

Figure:
schematic of target vs. target / target vs. reference decoding?
scatter plot target vs. target for active / passive
regression results showing contribution of noise variance / signal magnitude to delta dprime
"""
import helpers
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob
import statsmodels.api as sm
from settings import DIR
import scipy.stats as ss
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.size'] = 8

np.random.seed(123)

pr = False

if pr:
    df = pd.read_pickle(DIR+"results/res_pr.pickle")
    f1name = DIR + 'pyfigures/target_decoding_pr.svg'
else:
    df = pd.read_pickle(DIR+"results/res.pickle")
    f1name = DIR + 'pyfigures/target_decoding.svg'

dp_metric = 'dp_opt_sqrt'
s = 15
cmap = {
    'tar_tar': 'coral',
    'ref_ref': 'forestgreen',
    'tar_cat': 'black',
    'noise': 'tab:orange',
    'signal': 'tab:blue'
}
    

df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])

df['lambda_tot'] = df['evals'].apply(lambda x: sum(x))
df['dU_mag'] = df['dU'].apply(lambda x: np.linalg.norm(x))

rmask = ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & (df.ref_ref) & \
                (df.batch!=307) & (np.abs(np.log2(df.f1/df.f2))>0.3) & (np.abs(np.log2(df.f1/df.f2))<0.5)
tmask = ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.batch.isin([302, 307, 324]) & (df.tar_tar) & \
                (np.abs(np.log2(df.f1/df.f2))<=0.2) & (np.abs(np.log2(df.f1/df.f2))<=0.2)
mask = rmask | tmask
f = plt.figure(figsize=(4, 2))
ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 4), (0, 2), colspan=1)
ax3 = plt.subplot2grid((1, 4), (0, 3), colspan=1)

ax1.scatter(df[mask & ~df.active].groupby(by='site').mean()[dp_metric], 
                 df[mask & df.active].groupby(by='site').mean()[dp_metric], s=s, edgecolor='white', color=cmap['tar_tar'])
ax1.set_xlabel('Passive')
ax1.set_ylabel('Active')
mi = np.min(ax1.get_xlim()+ax1.get_ylim())
ma = np.max(ax1.get_xlim()+ax1.get_ylim())
ax1.plot([mi, ma], [mi, ma], '--', color='grey')

# statistical test of active vs. passive
d = {s: df[mask & df.active & (df.site==s)][dp_metric].values - df[mask & ~df.active & (df.site==s)][dp_metric].values for s in df[mask].site.unique()}
bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
p = get_direct_prob(bootsamp, np.zeros(len(bootsamp)))[0]

print(f"Active vs. Passive target discriminability, \n \
                Active: {df[mask & df.active][dp_metric].mean()}, Passive: {df[mask & ~df.active][dp_metric].mean()} \
                    pval: {p}, bootstrap test")
stat, p = ss.wilcoxon(df[mask & ~df.active].groupby(by='site').mean()[dp_metric], df[mask & df.active].groupby(by='site').mean()[dp_metric])
print(f"Active vs. Passive target discriminability, \n \
                Active: {df[mask & df.active].groupby(by='site').mean()[dp_metric].mean()}, Passive: {df[mask & ~df.active].groupby(by='site').mean()[dp_metric].mean()} \
                    pval: {p}, W: {stat} wilcoxon test")                    

# regression test to determine if noise or signal explains changes in target discriminability
X = pd.concat([df[mask & df.active]['dU_mag'] - df[mask & ~df.active]['dU_mag'],
              (df[mask & df.active]['lambda_tot'] - df[mask & ~df.active]['lambda_tot'])], axis=1)
X -= X.mean(axis=0)
X /= X.std(axis=0)
X = sm.add_constant(X)
y = (df[mask & df.active][dp_metric] - df[mask & ~df.active][dp_metric])
results = helpers.fit_OLS_model(X, y, replace=False, nboot=100, njacks=10)

print(results)

# plot regression coeff. / 95% confidence interval 
xerr = [results['coef']['lambda_tot'] - results['ci_coef']['lambda_tot'][0], \
                results['ci_coef']['lambda_tot'][1] - results['coef']['lambda_tot']]
ax2.errorbar([0.25], [results['coef']['lambda_tot']], marker='o', 
            yerr=np.array([xerr]).T, capsize=2, capthick=1, elinewidth=1, lw=0, color=cmap['noise'])
xerr = [results['coef']['dU_mag'] - results['ci_coef']['dU_mag'][0], \
                results['ci_coef']['dU_mag'][1] - results['coef']['dU_mag']]
ax2.errorbar([-0.25], results['coef']['dU_mag'], marker='o', 
            yerr=np.array([xerr]).T, capsize=2, capthick=1, elinewidth=1, lw=0, color=cmap['signal'])

ax2.axhline(0, linestyle='--', color='grey')
ax2.set_xlim((-0.5, 0.5))
ax2.set_ylabel(r"$cv\beta$ weight")


# plot unique variance explained by each predictor
xerr = [results['r2']['ulambda_tot'] - results['ci']['ulambda_tot'][0], \
                results['ci']['ulambda_tot'][1] - results['r2']['ulambda_tot']]
ax3.bar([0.5], [results['r2']['ulambda_tot']], 
            yerr=np.array([xerr]).T, error_kw=dict(capsize=2, capthick=1, elinewidth=1), edgecolor='k', lw=1, color=cmap['noise'], width=0.75)
xerr = [results['r2']['udU_mag'] - results['ci']['udU_mag'][0], \
                results['ci']['udU_mag'][1] - results['r2']['udU_mag']]
ax3.bar([-0.5], results['r2']['udU_mag'],  
            yerr=np.array([xerr]).T, error_kw=dict(capsize=2, capthick=1, elinewidth=1), edgecolor='k', lw=1, color=cmap['signal'], width=0.75)

ax3.set_xlim((-1, 1))
ax3.axhline(0, linestyle='--', color='grey')
ax3.set_ylabel(r'$cvR_{unique}^2$')
#ax3.set_ylim((-0.05, 0.2))

f.tight_layout()

f.savefig(f1name)

plt.show()