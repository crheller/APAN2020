"""
Show changes in dU vs. changes in noise variance.
    * Point is that both are striking, both predict changes in dprime, only one correlated with behavior
    * Also show results for behavior regression
 x 4 plot
"""
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
    
df = pd.read_pickle(DIR+"results/res_pr.pickle")
df['dp_opt_sqrt'] = np.sqrt(df['dp_opt'])
df['dp_diag_sqrt'] = np.sqrt(df['dp_diag'])
dfbp = pd.read_pickle(DIR+"results/res_pr_br.pickle")
dfbp['dp_opt_sqrt'] = np.sqrt(dfbp['dp_opt'])
dfbp['dp_diag_sqrt'] = np.sqrt(dfbp['dp_diag'])

di_metric = 'DI'  # for this data, DI = DIref if df.aref_tar = True
dp_metric = 'dp_opt_sqrt'
s = 50
cmap = {
    'signal': 'tab:blue',
    'noise': 'tab:orange',
    'raw': 'grey',
    'corrected': 'orchid'
}

mask = ~df.tdr_overall & ~df.pca & df.tdr_fixedNoise & df.aref_tar & df.batch.isin([302, 307, 324])
df = df[mask]
dfbp = dfbp[mask]

df['lambda_tot'] = df['evals'].apply(lambda x: sum(x))
df['dU_mag'] = df['dU'].apply(lambda x: np.linalg.norm(x))
dfbp['lambda_tot'] = dfbp['evals'].apply(lambda x: sum(x))
dfbp['dU_mag'] = dfbp['dU'].apply(lambda x: np.linalg.norm(x))


f, ax = plt.subplots(2, 4, figsize=(16, 8))

# ======================= first order changes ===============================
ax[0, 0].scatter(df[~df.active]['dU_mag'], df[df.active]['dU_mag'], s=s, edgecolor='white', color=cmap['signal'])
ax[0, 0].set_xlabel("Passive")
ax[0, 0].set_ylabel("Active")
ax[0, 0].set_title("Signal magnitude")

ax[1, 0].scatter(dfbp[~df.active]['dU_mag'], dfbp[df.active]['dU_mag'], s=s, edgecolor='white', color=cmap['signal'])
ax[1, 0].set_xlabel("Passive")
ax[1, 0].set_ylabel("Active")
ax[1, 0].set_title("Signal magnitude")


m = np.max(ax[0, 0].get_xlim()+ax[0, 0].get_ylim()+ax[1, 0].get_xlim()+ax[1, 0].get_ylim())
mi = np.min(ax[0, 0].get_xlim()+ax[0, 0].get_ylim()+ax[1, 0].get_xlim()+ax[1, 0].get_ylim())
ax[0, 0].plot([mi, m], [mi, m], '--', color='grey')
ax[1, 0].plot([mi, m], [mi, m], '--', color='grey')

# ===================== second order changes ==============================
ax[0, 1].scatter(df[~df.active]['lambda_tot'], df[df.active]['lambda_tot'], s=s, edgecolor='white', color=cmap['noise'])
ax[0, 1].set_xlabel("Passive")
ax[0, 1].set_ylabel("Active")
ax[0, 1].set_title("Shared noise variance")

ax[1, 1].scatter(dfbp[~df.active]['lambda_tot'], dfbp[df.active]['lambda_tot'], s=s, edgecolor='white', color=cmap['noise'])
ax[1, 1].set_xlabel("Passive")
ax[1, 1].set_ylabel("Active")
ax[1, 1].set_title("Shared noise variance")


m = np.max(ax[0, 1].get_xlim()+ax[0, 1].get_ylim()+ax[1, 1].get_xlim()+ax[1, 1].get_ylim())
mi = np.min(ax[0, 1].get_xlim()+ax[0, 1].get_ylim()+ax[1, 1].get_xlim()+ax[1, 1].get_ylim())
ax[0, 1].plot([mi, m], [mi, m], '--', color='grey')
ax[1, 1].plot([mi, m], [mi, m], '--', color='grey')

# ========================== REGRESSION MODELS ===========================
# predict delta dprime from noise / signal for the raw and corrected data
X = pd.concat([df[df.active]['dU_mag'] - df[~df.active]['dU_mag'],
              (df[df.active]['lambda_tot'] - df[~df.active]['lambda_tot'])], axis=1)
X -= X.mean(axis=0)
X /= X.std(axis=0)
X = sm.add_constant(X)
y = (df[df.active][dp_metric] - df[~df.active][dp_metric])
results = helpers.fit_OLS_model(X, y, replace=False, nboot=100, njacks=5)

Xbp = pd.concat([dfbp[df.active]['dU_mag'] - dfbp[~df.active]['dU_mag'],
              dfbp[df.active]['lambda_tot'] - dfbp[~df.active]['lambda_tot']], axis=1)
Xbp -= Xbp.mean(axis=0)
Xbp /= Xbp.std(axis=0)
Xbp = sm.add_constant(Xbp)
ybp = dfbp[df.active][dp_metric] - dfbp[~df.active][dp_metric]
results_bp = helpers.fit_OLS_model(Xbp, ybp, replace=False, nboot=100, njacks=5)

# plot regression coeff. / 95% confidence interval 
xerr = [results['coef']['lambda_tot'] - results['ci_coef']['lambda_tot'][0], \
                results['ci_coef']['lambda_tot'][1] - results['coef']['lambda_tot']]
ax[1, 2].errorbar([-0.25], [results['coef']['lambda_tot']], marker='o', 
            yerr=np.array([xerr]).T, capsize=3, capthick=2, elinewidth=2, lw=0, color=cmap['noise'])
xerr = [results_bp['coef']['lambda_tot'] - results_bp['ci_coef']['lambda_tot'][0], \
                results_bp['ci_coef']['lambda_tot'][1] - results_bp['coef']['lambda_tot']]
ax[1, 2].errorbar([1], results_bp['coef']['lambda_tot'], marker='o', 
            yerr=np.array([xerr]).T, capsize=3, capthick=2, elinewidth=2, lw=0, color=cmap['noise'])


xerr = [results['coef']['dU_mag'] - results['ci_coef']['dU_mag'][0], \
                results['ci_coef']['dU_mag'][1] - results['coef']['dU_mag']]
ax[1, 2].errorbar([0], results['coef']['dU_mag'], marker='o', 
            yerr=np.array([xerr]).T, capsize=3, capthick=2, elinewidth=2, lw=0, color=cmap['signal'])
xerr = [results_bp['coef']['dU_mag'] - results_bp['ci_coef']['dU_mag'][0], \
                results_bp['ci_coef']['dU_mag'][1] - results_bp['coef']['dU_mag']]
ax[1, 2].errorbar([1.25], results_bp['coef']['dU_mag'], marker='o', 
            yerr=np.array([xerr]).T, capsize=3, capthick=2, elinewidth=2, lw=0, color=cmap['signal'])

ax[1, 2].set_title(r"$\Delta d'$ Regression Coefficients")
ax[1, 2].axhline(0, linestyle='--', color='grey')
ax[1, 2].set_xlim((-2, 3))
ax[1, 2].set_ylabel('Cross-validated\nregression weight')
ax[1, 2].set_xticks([-0.5, 0.25, 0.75, 1.5])
ax[1, 2].set_xticklabels([r"$\Delta$ Shared"+" noise variance",
                         r"$\Delta$ Signal"+" magnitude",
                         r"$\Delta$ Shared"+" noise variance",
                         r"$\Delta$ Signal"+" magnitude"], rotation=45)

# plot unique variance explained by each predictor
xerr = [results['r2']['ulambda_tot'] - results['ci']['ulambda_tot'][0], \
                results['ci']['ulambda_tot'][1] - results['r2']['ulambda_tot']]
ax[0, 2].bar([-0.5], [results['r2']['ulambda_tot']], 
            yerr=np.array([xerr]).T, error_kw=dict(capsize=3, capthick=2, elinewidth=2), edgecolor='k', lw=2, color=cmap['noise'], width=0.5)
xerr = [results_bp['r2']['ulambda_tot'] - results_bp['ci']['ulambda_tot'][0], \
                results_bp['ci']['ulambda_tot'][1] - results_bp['r2']['ulambda_tot']]
ax[0, 2].bar([1], results_bp['r2']['ulambda_tot'],  
            yerr=np.array([xerr]).T, error_kw=dict(capsize=3, capthick=2, elinewidth=2), edgecolor='k', lw=2, color=cmap['noise'], width=0.5)


xerr = [results['r2']['udU_mag'] - results['ci']['udU_mag'][0], \
                results['ci']['udU_mag'][1] - results['r2']['udU_mag']]
ax[0, 2].bar([0], results['r2']['udU_mag'],
            yerr=np.array([xerr]).T, error_kw=dict(capsize=3, capthick=2, elinewidth=2), edgecolor='k', lw=2, color=cmap['signal'], width=0.5)
xerr = [results_bp['r2']['udU_mag'] - results_bp['ci']['udU_mag'][0], \
                results_bp['ci']['udU_mag'][1] - results_bp['r2']['udU_mag']]
ax[0, 2].bar([1.5], results_bp['r2']['udU_mag'], 
            yerr=np.array([xerr]).T, error_kw=dict(capsize=3, capthick=2, elinewidth=2), edgecolor='k', lw=2, color=cmap['signal'], width=0.5)

ax[0, 2].set_xlim((-2, 3))
ax[0, 2].set_title(r"$\Delta d'$ Explained Variance")
ax[0, 2].axhline(0, linestyle='--', color='grey')
ax[0, 2].set_ylabel(r'$cvR_{unique}^2$')
ax[0, 2].set_ylim((-0.05, 0.4))


# ========================== Model behavior as fn of noise / signal changes =============================
X = pd.concat([df[df.active]['dU_mag'] - df[~df.active]['dU_mag'],
              (df[df.active]['lambda_tot'] - df[~df.active]['lambda_tot'])], axis=1)
X = sm.add_constant(X)
y = df[df.active][di_metric]
r = helpers.fit_OLS_model(X, y, replace=False, nboot=100, njacks=5)

# plot relationship between behavior / noise and behavior / signal
sns.regplot(x=y, y=X['dU_mag'], ax=ax[0, 3], color=cmap['signal'])
ax[0, 3].axhline(0, linestyle='--', color='grey')
ax[0, 3].axvline(0.5, linestyle='--', color='grey')
ax[0, 3].set_ylabel(r"$\Delta$ Signal magnitude")
ax[0, 3].set_xlabel('Behavior performance (DI')
ax[0, 3].set_title(r"$R^2$: %s, $p > 0.05$" % round(r['r2']['dU_mag'], 3))

sns.regplot(x=y, y=X['lambda_tot'], ax=ax[1, 3], color=cmap['noise'])
ax[1, 3].set_ylabel(r"$\Delta$ Shared noise variance")
ax[1, 3].set_xlabel('Behavior performance (DI')
ax[1, 3].axhline(0, linestyle='--', color='grey')
ax[1, 3].axvline(0.5, linestyle='--', color='grey')
ax[1, 3].set_title(r"$R^2$: %s, $p < 0.05$" % round(r['r2']['lambda_tot'], 3))

f.tight_layout()

f.savefig(DIR + 'pyfigures/first_vs_second_order.svg')

plt.show()



