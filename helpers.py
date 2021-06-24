import statsmodels.api as sm
import numpy as np

np.random.seed(123)
def fit_OLS_model(X, y, replace=True, nboot=100, njacks=10):
    """
    Compute bootstrapped estimates of cross-validated R2 for the full model,
    and unique R2s for each predictor

    X and y are dataframes / series. Use statsmodels to fit
    """
    reg = [k for k in X.columns if k!='const']
    if 'const' in X.columns: const=['const'] 
    else: const=[]
    r2 = []
    r2i = {k: [] for k in reg}
    r2u = {k: [] for k in reg}
    coef = {k: [] for k in reg}
    # n-fold cross-val for each of 100 different bootstrap resamples
    for boot in range(nboot):
        inds = np.random.choice(range(0, X.shape[0]), X.shape[0], replace=replace).tolist()
        pred = []
        predshuf = {k: [] for k in reg}
        predind = {k: [] for k in reg}
        test_data = []
        for i in range(njacks):
            fidx = inds[:np.floor(i*X.shape[0]/njacks).astype(np.int)] + inds[np.floor((i+1)*X.shape[0]/njacks).astype(np.int):]
            idx = inds[np.floor(i*X.shape[0]/njacks).astype(np.int):np.floor((i+1)*X.shape[0]/njacks).astype(np.int)]
            xfit = X.iloc[fidx, :]
            yfit = y.iloc[fidx]
            xtest = X.iloc[idx, :]
            ytest = y.iloc[idx]

            # raw data
            m = sm.OLS(yfit, xfit).fit()
            pred.append(m.predict(xtest).values)
            test_data.append(ytest.values)
            for k in reg:
                coef[k].append(m.params[k])

            # shuffle params
            for k in reg:
                xshuf1 = xfit.copy()
                xshuf1[k] = np.random.permutation(xshuf1[k].values)
                ms1 = sm.OLS(yfit, xshuf1).fit()
                xshuf1test = xtest.copy()
                xshuf1test[k] = np.random.permutation(xshuf1test[k].values)
                predshuf[k].append(ms1.predict(xshuf1test).values)
            
            # model with single predictor
            for k in reg:
                xind = xfit.copy()
                xind = xind[[k]+const]
                mi = sm.OLS(yfit, xind).fit()
                xindtest = xtest.copy()
                xindtest = xindtest[[k]+const]
                predind[k].append(mi.predict(xindtest).values)

        # compute cross-validated r2 for full model
        _r2 = (1 - (sum((np.concatenate(pred) - np.concatenate(test_data))**2) / \
                        sum((np.concatenate(test_data) - np.concatenate(test_data).mean())**2))) 
        r2.append(_r2)

        # single regressor r2s
        for k in reg:
            _r2i = (1 - (sum((np.concatenate(predind[k]) - np.concatenate(test_data))**2) / \
                        sum((np.concatenate(test_data) - np.concatenate(test_data).mean())**2))) 
            r2i[k].append(_r2i)
        
        # get unique r2s from shuffling
        for k in reg:
            _r2shuf = (1 - (sum((np.concatenate(predshuf[k]) - np.concatenate(test_data))**2) / \
                        sum((np.concatenate(test_data) - np.concatenate(test_data).mean())**2))) 
            _r2u = _r2 - _r2shuf
            if _r2shuf > 0: r2u[k].append(_r2u) 
            else: r2u[k].append(_r2)

    # compute mean / 95% CI, pack into results dict
    results = {'r2': {}, 'ci': {}, 'coef': {}, 'ci_coef': {}}
    results['r2']['full'] = np.mean(r2)
    results['ci']['full'] = (np.quantile(r2, 0.025), np.quantile(r2, 0.975))

    for k in reg:
        results['r2']['u'+k] = np.mean(r2u[k])
        results['ci']['u'+k] = (np.quantile(r2u[k], 0.025), np.quantile(r2u[k], 0.975))
        results['r2'][k] = np.mean(r2i[k])
        results['ci'][k] = (np.quantile(r2i[k], 0.025), np.quantile(r2i[k], 0.975))
        results['coef'][k] = np.mean(coef[k])
        results['ci_coef'][k] =  (np.quantile(coef[k], 0.025), np.quantile(coef[k], 0.975))

    return results

