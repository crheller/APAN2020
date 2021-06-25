"""
Created for dissertation -- needed to resize / format for MS since the other script was made for poster

Make two figures
1.1 signal mag scatter
1.2 noise var scatter
1.3 regression results
(in a row)

2.1 delta signal vs. behavior
2.2 delta noise vs. behavior
(stacked on top of each other)
"""
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob
import helpers
from settings import DIR
import scipy.stats as ss
import statsmodels.api as sm
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
    df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
    df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
    f1name = DIR + 'pyfigures/first_vs_second_order1_pr.svg'
    f2name = DIR + 'pyfigures/first_vs_second_order2_pr.svg'
else:
    df = pd.read_pickle(DIR+"results/res.pickle")
    df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
    df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
    f1name = DIR + 'pyfigures/first_vs_second_order1.svg'
    f2name = DIR + 'pyfigures/first_vs_second_order2.svg'

di_metric = 'DI'  # for this data, DI = DIref if df.aref_tar = True
dp_metric = 'dp_opt_sqrt'
s = 15
cmap = {
    'signal': 'tab:blue',
    'noise': 'tab:orange',
    'raw': 'grey',
    'corrected': 'orchid'
}

mask = ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.aref_tar & df.batch.isin([302, 307, 324])
df = df[mask]

df['lambda_tot'] = df['evals'].apply(lambda x: sum(x))
df['dU_mag'] = df['dU'].apply(lambda x: np.linalg.norm(x))

# FIGURE ONE
f = plt.figure(figsize=(6, 2))
ax1 = plt.subplot2grid((1, 6), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 6), (0, 2), colspan=2)
ax3 = plt.subplot2grid((1, 6), (0, 4), colspan=1)
ax4 = plt.subplot2grid((1, 6), (0, 5), colspan=1)
# ======================= first order changes ===============================
ax1.scatter(df[~df.active]['dU_mag'], df[df.active]['dU_mag'], s=s, edgecolor='white', color=cmap['signal'])
ax1.set_xlabel("Passive")
ax1.set_ylabel("Active")
ax1.set_title("Signal magnitude")
d = {s: df[(df.site==s) & df.active]['dU_mag'].values - 
                    df[(df.site==s) & ~df.active]['dU_mag'].values for s in df.site.unique()}
bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
p = get_direct_prob(bootsamp, np.zeros(len(bootsamp)))[0]
print(r"Target vs. Reference signal mag"+
        f"\n active: {round(df[df.active]['dU_mag'].mean().astype(float), 3)}, passive: {round(df[~df.active]['dU_mag'].mean(), 3)}, pval: {round(p, 3)}")

# ========================= second order changes ============================
ax2.scatter(df[~df.active]['lambda_tot'], df[df.active]['lambda_tot'], s=s, edgecolor='white', color=cmap['noise'])
ax2.set_xlabel("Passive")
ax2.set_ylabel("Active")
ax2.set_title("Shared noise variance")
d = {s: df[(df.site==s) & ~df.active]['lambda_tot'].values - 
                    df[(df.site==s) & df.active]['lambda_tot'].values for s in df.site.unique()}
bootsamp = get_bootstrapped_sample(d, metric='mean', even_sample=False, nboot=1000)
p = get_direct_prob(bootsamp, np.zeros(len(bootsamp)))[0]
print(r"Target vs. Reference Discriminability noise var"+
        f"\n active: {round(df[df.active]['lambda_tot'].mean().astype(float), 3)}, passive: {round(df[~df.active]['lambda_tot'].mean(), 3)}, pval: {round(p, 3)}")

# plot unity lines
m = np.max(ax1.get_xlim()+ax1.get_ylim())
mi = np.min(ax1.get_xlim()+ax1.get_ylim())
ax1.plot([mi, m], [mi, m], '--', color='grey')
m = np.max(ax2.get_xlim()+ax2.get_ylim())
mi = np.min(ax2.get_xlim()+ax2.get_ylim())
ax2.plot([mi, m], [mi, m], '--', color='grey')

# ========================== Regression results ============================
# predict delta dprime from noise / signal for the raw and corrected data
X = pd.concat([df[df.active]['dU_mag'] - df[~df.active]['dU_mag'],
              (df[df.active]['lambda_tot'] - df[~df.active]['lambda_tot'])], axis=1)
X -= X.mean(axis=0)
X /= X.std(axis=0)
X = sm.add_constant(X)
y = (df[df.active][dp_metric] - df[~df.active][dp_metric])
results = helpers.fit_OLS_model(X, y, replace=False, nboot=100, njacks=5)

# plot regression coeff. / 95% confidence interval 
xerr = [results['coef']['lambda_tot'] - results['ci_coef']['lambda_tot'][0], \
                results['ci_coef']['lambda_tot'][1] - results['coef']['lambda_tot']]
ax3.errorbar([0.25], [results['coef']['lambda_tot']], marker='o', 
            yerr=np.array([xerr]).T, capsize=2, capthick=1, elinewidth=1, lw=0, color=cmap['noise'])
xerr = [results['coef']['dU_mag'] - results['ci_coef']['dU_mag'][0], \
                results['ci_coef']['dU_mag'][1] - results['coef']['dU_mag']]
ax3.errorbar([-0.25], results['coef']['dU_mag'], marker='o', 
            yerr=np.array([xerr]).T, capsize=2, capthick=1, elinewidth=1, lw=0, color=cmap['signal'])

ax3.axhline(0, linestyle='--', color='grey')
ax3.set_xlim((-0.5, 0.5))
ax3.set_ylabel(r"$cv\beta$ weight")


# plot unique variance explained by each predictor
xerr = [results['r2']['ulambda_tot'] - results['ci']['ulambda_tot'][0], \
                results['ci']['ulambda_tot'][1] - results['r2']['ulambda_tot']]
ax4.bar([0.5], [results['r2']['ulambda_tot']], 
            yerr=np.array([xerr]).T, error_kw=dict(capsize=2, capthick=1, elinewidth=1), edgecolor='k', lw=1, color=cmap['noise'], width=0.5)
xerr = [results['r2']['udU_mag'] - results['ci']['udU_mag'][0], \
                results['ci']['udU_mag'][1] - results['r2']['udU_mag']]
ax4.bar([-0.5], results['r2']['udU_mag'],  
            yerr=np.array([xerr]).T, error_kw=dict(capsize=2, capthick=1, elinewidth=1), edgecolor='k', lw=1, color=cmap['signal'], width=0.5)

ax4.set_xlim((-1, 1))
ax4.axhline(0, linestyle='--', color='grey')
ax4.set_ylabel(r'$cvR_{unique}^2$')
ax4.set_ylim((-0.05, 0.4))

f.tight_layout()

print("Delta dprime regression: ")
print(results)

f.savefig(f1name)


# ========================== Model behavior as fn of noise / signal changes =============================
X = pd.concat([df[df.active]['dU_mag'] - df[~df.active]['dU_mag'],
              (df[df.active]['lambda_tot'] - df[~df.active]['lambda_tot'])], axis=1)
X = sm.add_constant(X)
y = df[df.active][di_metric]
r = helpers.fit_OLS_model(X, y, replace=False, nboot=100, njacks=5)

f, ax = plt.subplots(2, 1, figsize=(2.3, 4), sharex=True)

# plot relationship between behavior / noise and behavior / signal
sns.regplot(x=y, y=X['dU_mag'], ax=ax[0], color=cmap['signal'], scatter_kws={'s': s})
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].axvline(0.5, linestyle='--', color='grey')
ax[0].set_ylabel(r"$\Delta$ Signal magnitude")
ax[0].set_title(r"$R^2$: %s, $p > 0.05$" % round(r['r2']['dU_mag'], 3))

sns.regplot(x=y, y=X['lambda_tot'], ax=ax[1], color=cmap['noise'], scatter_kws={'s': s})
ax[1].set_ylabel(r"$\Delta$ Shared noise variance")
ax[1].set_xlabel('Behavior performance (DI)')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0.5, linestyle='--', color='grey')
ax[1].set_title(r"$R^2$: %s, $p < 0.05$" % round(r['r2']['lambda_tot'], 3))

print("Behavior regression: ")
print(r)

f.tight_layout()

f.savefig(f2name)

plt.show()



