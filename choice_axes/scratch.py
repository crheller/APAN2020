"""
For an example site, look at relationship between choice axis(es), 
decoding axis, and noise correlation axis

Use leave-one-out subsampling of neurons to generate prob. distribution 
over angles between vectors, and compare to distribution of random vectors 
in that nD space to see if significantly more/less aligned than chance.
"""
from settings import DIR
from sklearn.decomposition import PCA
from nems_lbhb.baphy_experiment import BAPHYExperiment
import numpy as np
import pandas as pd
import pickle
from itertools import combinations, product
from charlieTools.decoding import compute_dprime
import nems_lbhb.tin_helpers as thelp
import nems.db as nd

batches = [324, 325]
rawid = None
recache = False
options = {'resp': True, 'pupil': True, 'rasterfs': 10}

tstart = int( 0.1 * options['rasterfs'] )
tend = int( 0.3 * options['rasterfs'] )

choice_decoder = pd.DataFrame()
stimulus_decoder = pd.DataFrame()
for batch in batches:
    sites = np.unique([s[:7] for s in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if (s != 'ARM007c') & (s != 'CRD013b')]
    for site in sites:
        manager = BAPHYExperiment(batch=batch, siteid=site[:7], rawid=rawid)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()

        # ================== here is where we'd pick a subset of neurons, if we wanted to ====================
        nNeurons = rec['resp'].shape[0]

        axes = {}
        # ===================== GET CATCH CHOICE AXIS =======================
        cat_mask = ['CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL']
        rcat = rec.copy()
        rcat = rcat.and_mask(cat_mask)
        rcat['cat_choice'] = rcat['resp'].epoch_to_signal('CORRECT_REJECT_TRIAL')
        catches = [s for s in rcat.apply_mask(reset_epochs=True).epochs.name.unique() if 'CAT_' in s]

        # define choice axis
        d = np.concatenate([(rcat['resp'].extract_epoch(c, mask=rcat['mask'])[:, :, tstart:tend].mean(axis=-1) - \
                            rcat['resp'].extract_epoch(c, mask=rcat['mask'])[:, :, tstart:tend].mean(axis=(0, -1))) for c in catches], 
                            axis=0)
        c = np.concatenate([rcat['cat_choice'].extract_epoch(t, mask=rcat['mask'])[:, 0, 0] for t in catches])
        cat_choice = d[c].mean(axis=0) - d[~c].mean(axis=0)
        cat_choice /= np.linalg.norm(cat_choice)
        # deflate out choice axis, and measure the remaining PCs
        r1 = d.dot(cat_choice)[:, np.newaxis].dot(cat_choice[np.newaxis])
        d -= r1
        pca = PCA(n_components=np.linalg.matrix_rank(d))
        pca.fit(d)
        axes['catch'] = np.concatenate([cat_choice[np.newaxis], pca.components_], axis=0)

        # ===================== GET TARGET CHOICE AXIS =======================
        tar_mask = ['HIT_TRIAL', 'MISS_TRIAL']
        rtar = rec.copy()
        rtar = rtar.and_mask(tar_mask)
        rtar['tar_choice'] = rtar['resp'].epoch_to_signal('HIT_TRIAL')
        targets = [s for s in rtar.apply_mask(reset_epochs=True).epochs.name.unique() if 'TAR_' in s]

        # define choice axis
        d = np.concatenate([(rtar['resp'].extract_epoch(t, mask=rtar['mask'])[:, :, tstart:tend].mean(axis=-1) - \
                            rtar['resp'].extract_epoch(t, mask=rtar['mask'])[:, :, tstart:tend].mean(axis=(0, -1))) for t in targets], 
                            axis=0)
        c = np.concatenate([rtar['tar_choice'].extract_epoch(t, mask=rtar['mask'])[:, 0, 0] for t in targets])
        if sum(c==False) > 5:
            # need at least 5 misses
            tar_choice = d[c].mean(axis=0) - d[~c].mean(axis=0)
            tar_choice /= np.linalg.norm(tar_choice)
            r1 = d.dot(tar_choice)[:, np.newaxis].dot(tar_choice[np.newaxis])
            d -= r1
            pca = PCA(n_components=np.linalg.matrix_rank(d))
            pca.fit(d)
            axes['target'] = np.concatenate([tar_choice[np.newaxis], pca.components_], axis=0)
        else:
            axes['target'] = np.nan

        # ======================= GET PC NOISE AXIS ===========================
        rp = rec.and_mask(['PASSIVE_EXPERIMENT'])
        d = np.concatenate([(rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, tstart:tend].mean(axis=-1) - \
                            rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, tstart:tend].mean(axis=(0, -1))) for t in targets+catches], 
                            axis=0)
        pca = PCA(n_components=nNeurons)
        pca.fit(d)
        axes['pca'] = pca.components_

        # ======================= GET DELTA NOISE AXIS ==========================
        lv = pickle.load(open(DIR + '/results/drsc_axes.pickle', "rb"))
        axes['delta'] = lv[site]['tarCat']['evecs']

        # ================= GET CATCH VS. TARGET STIMULUS AXIS ============================
        dmc = np.concatenate([rcat['resp'].extract_epoch(c, mask=rcat['mask'])[:, :, tstart:tend].mean(axis=(0, -1))[np.newaxis, :] for c in catches], axis=0).mean(axis=0)
        dmt = np.concatenate([rtar['resp'].extract_epoch(c, mask=rtar['mask'])[:, :, tstart:tend].mean(axis=(0, -1))[np.newaxis, :] for c in targets], axis=0).mean(axis=0)
        tc_axis = (dmt - dmc) / np.linalg.norm(dmt - dmc)
        r1 = d.dot(tc_axis)[:, np.newaxis].dot(tc_axis[np.newaxis])
        d -= r1
        pca = PCA(n_components=np.linalg.matrix_rank(d))
        pca.fit(d)
        axes['tarCat'] = np.concatenate((tc_axis[np.newaxis, :], pca.components_), axis=0)

        nPCs = min([8, rec['resp'].shape[0]])
        # ============================= CHOICE DECODING =========================
        # for each stimulus, decode choice using the above sets of axes. Use leave-one-out
        # decoding and report the proportion correct
        d = np.concatenate([(rcat['resp'].extract_epoch(c, mask=rcat['mask'])[:, :, tstart:tend].mean(axis=-1) - \
                            rcat['resp'].extract_epoch(c, mask=rcat['mask'])[:, :, tstart:tend].mean(axis=(0, -1))) for c in catches], 
                            axis=0)
        choice = np.concatenate([rcat['cat_choice'].extract_epoch(t, mask=rcat['mask'])[:, 0, 0] for t in catches])
        for nAx in range(nPCs): 
            cat_correct = np.zeros(d.shape[0])
            tar_correct = np.zeros(d.shape[0])
            pca_correct = np.zeros(d.shape[0])
            delt_correct = np.zeros(d.shape[0])
            tc_correct = np.zeros(d.shape[0])
            cat_mask = np.zeros(d.shape[0]).astype(bool)
            tar_mask = np.zeros(d.shape[0]).astype(bool)
            pca_mask = np.zeros(d.shape[0]).astype(bool)
            delt_mask = np.zeros(d.shape[0]).astype(bool)
            tc_mask = np.zeros(d.shape[0]).astype(bool)
            for i in range(d.shape[0]):
                idx = np.array(list(set(range(d.shape[0])).difference(set([i]))))
                for ax_str in ['catch', 'target', 'pca', 'delta', 'tarCat']:
                    _d = d[idx]
                    _choice = choice[idx]
                    # project into space
                    _d = _d.dot(axes[ax_str][range(0, nAx+1), :].T)
                    # compute discrimination axis
                    if nAx > 0:
                        _, wopt, _, _, _, _= compute_dprime(_d[_choice].T, _d[~_choice].T)
                        _d = _d.dot(wopt / np.linalg.norm(wopt))
                    
                        # project held out point onto axis
                        val = d[[i], :].dot(axes[ax_str][range(0, nAx+1), :].T).dot(wopt / np.linalg.norm(wopt))[0][0]
                    else:
                        val = d[[i], :].dot(axes[ax_str][range(0, nAx+1), :].T)
                    # save projections            
                    if ax_str=='catch':
                        cat_correct[i] = val
                        cat_mask[i] = choice[i]
                    elif ax_str=='target': 
                        tar_correct[i] = val
                        tar_mask[i] = choice[i]
                    elif ax_str=='pca':
                        pca_correct[i] = val
                        pca_mask[i] = choice[i]
                    elif ax_str=='delta':
                        delt_correct[i] = val
                        delt_mask[i] = choice[i]
                    elif ax_str=='tarCat':
                        tc_correct[i] = val
                        tc_mask[i] = choice[i]
            
            # compute dprime for each 
            cdp, _, _, _, _, _ = compute_dprime(cat_correct[cat_mask][np.newaxis, :], cat_correct[~cat_mask][np.newaxis, :])
            tdp, _, _, _, _, _ = compute_dprime(tar_correct[tar_mask][np.newaxis, :], tar_correct[~tar_mask][np.newaxis, :])
            pdp, _, _, _, _, _ = compute_dprime(pca_correct[pca_mask][np.newaxis, :], pca_correct[~pca_mask][np.newaxis, :])
            ddp, _, _, _, _, _ = compute_dprime(delt_correct[delt_mask][np.newaxis, :], delt_correct[~delt_mask][np.newaxis, :])
            tcdp, _, _, _, _, _ = compute_dprime(tc_correct[tc_mask][np.newaxis, :], tc_correct[~tc_mask][np.newaxis, :])

            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['catch', cdp** 0.5, nAx, 'catch', 'catch', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['catch', tdp** 0.5, nAx, 'target', 'catch', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['catch', pdp** 0.5, nAx, 'pca', 'catch', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['catch', ddp** 0.5, nAx, 'delta', 'catch', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['catch', tcdp** 0.5, nAx, 'tarCat', 'catch', site, batch]).T)

        # HIT / MISS DECODING
        d = np.concatenate([(rtar['resp'].extract_epoch(t, mask=rtar['mask'])[:, :, tstart:tend].mean(axis=-1) - \
                            rtar['resp'].extract_epoch(t, mask=rtar['mask'])[:, :, tstart:tend].mean(axis=(0, -1))) for t in targets], 
                            axis=0)
        choice = np.concatenate([rtar['tar_choice'].extract_epoch(t, mask=rtar['mask'])[:, 0, 0] for t in targets])
        for nAx in range(nPCs): 
            cat_correct = np.zeros(d.shape[0])
            tar_correct = np.zeros(d.shape[0])
            pca_correct = np.zeros(d.shape[0])
            delt_correct = np.zeros(d.shape[0])
            tc_correct = np.zeros(d.shape[0])
            cat_mask = np.zeros(d.shape[0]).astype(bool)
            tar_mask = np.zeros(d.shape[0]).astype(bool)
            pca_mask = np.zeros(d.shape[0]).astype(bool)
            delt_mask = np.zeros(d.shape[0]).astype(bool)
            tc_mask = np.zeros(d.shape[0]).astype(bool)
            for i in range(d.shape[0]):
                idx = np.array(list(set(range(d.shape[0])).difference(set([i]))))
                for ax_str in ['catch', 'target', 'pca', 'delta', 'tarCat']:
                    _d = d[idx]
                    _choice = choice[idx]
                    # project into space
                    _d = _d.dot(axes[ax_str][range(0, nAx+1), :].T)
                    # compute discrimination axis
                    if nAx > 0:
                        _, wopt, _, _, _, _= compute_dprime(_d[_choice].T, _d[~_choice].T)
                        _d = _d.dot(wopt / np.linalg.norm(wopt))
                    
                        # project held out point onto axis
                        val = d[[i], :].dot(axes[ax_str][range(0, nAx+1), :].T).dot(wopt / np.linalg.norm(wopt))[0][0]
                    else:
                        val = d[[i], :].dot(axes[ax_str][range(0, nAx+1), :].T)
                    # save projections            
                    if ax_str=='catch':
                        cat_correct[i] = val
                        cat_mask[i] = choice[i]
                    elif ax_str=='target': 
                        tar_correct[i] = val
                        tar_mask[i] = choice[i]
                    elif ax_str=='pca':
                        pca_correct[i] = val
                        pca_mask[i] = choice[i]
                    elif ax_str=='delta':
                        delt_correct[i] = val
                        delt_mask[i] = choice[i]
                    elif ax_str=='tarCat':
                        tc_correct[i] = val
                        tc_mask[i] = choice[i]
            
            # compute dprime for each 
            cdp, _, _, _, _, _ = compute_dprime(cat_correct[cat_mask][np.newaxis, :], cat_correct[~cat_mask][np.newaxis, :])
            tdp, _, _, _, _, _ = compute_dprime(tar_correct[tar_mask][np.newaxis, :], tar_correct[~tar_mask][np.newaxis, :])
            pdp, _, _, _, _, _ = compute_dprime(pca_correct[pca_mask][np.newaxis, :], pca_correct[~pca_mask][np.newaxis, :])
            ddp, _, _, _, _, _ = compute_dprime(delt_correct[delt_mask][np.newaxis, :], delt_correct[~delt_mask][np.newaxis, :])
            tcdp, _, _, _, _, _ = compute_dprime(tc_correct[tc_mask][np.newaxis, :], tc_correct[~tc_mask][np.newaxis, :])
            
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['target', cdp** 0.5, nAx, 'catch', 'target', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['target', tdp** 0.5, nAx, 'target', 'target', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['target', pdp** 0.5, nAx, 'pca', 'target', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['target', ddp** 0.5, nAx, 'delta', 'target', site, batch]).T)
            choice_decoder = choice_decoder.append(pd.DataFrame(index=['sound', 'dprime', 'nDim', 'axes', 'soundCategory', 'site', 'batch'],
                                    data=['target', tcdp** 0.5, nAx, 'tarCat', 'target', site, batch]).T)


        # ======================= STIMULUS DECODING ============================
        # perform stimulus decoding on each set of axes. 
        stim_pairs = list(product(catches, targets))
        for pair in stim_pairs:
            d1 = rcat['resp'].extract_epoch(pair[0], mask=rcat['mask'])[:, :, tstart:tend].mean(axis=-1)
            d2 = rtar['resp'].extract_epoch(pair[1], mask=rtar['mask'])[:, :, tstart:tend].mean(axis=-1)
            snr = thelp.get_snrs([pair[1]])[0]
            d = np.concatenate((d1, d2), axis=0)
            mask = np.concatenate((np.zeros(d1.shape[0]), np.ones(d2.shape[0])), axis=0).astype(bool)
            for nAx in range(nPCs): 
                cat_correct = np.zeros(d.shape[0])
                tar_correct = np.zeros(d.shape[0])
                pca_correct = np.zeros(d.shape[0])
                delt_correct = np.zeros(d.shape[0])
                tc_correct = np.zeros(d.shape[0])
                cat_mask = np.zeros(d.shape[0]).astype(bool)
                tar_mask = np.zeros(d.shape[0]).astype(bool)
                pca_mask = np.zeros(d.shape[0]).astype(bool)
                delt_mask = np.zeros(d.shape[0]).astype(bool)
                tc_mask = np.zeros(d.shape[0]).astype(bool)
                for i in range(d.shape[0]):
                    idx = np.array(list(set(range(d.shape[0])).difference(set([i]))))
                    for ax_str in ['catch', 'target', 'pca', 'delta', 'tarCat']:
                        _d = d[idx]
                        _choice = mask[idx]
                        # project into space
                        _d = _d.dot(axes[ax_str][range(0, nAx+1), :].T)
                        # compute discrimination axis
                        if nAx > 0:
                            _, wopt, _, _, _, _= compute_dprime(_d[_choice].T, _d[~_choice].T)
                            _d = _d.dot(wopt / np.linalg.norm(wopt))
                        
                            # project held out point onto axis
                            val = d[[i], :].dot(axes[ax_str][range(0, nAx+1), :].T).dot(wopt / np.linalg.norm(wopt))[0][0]
                        else:
                            val = d[[i], :].dot(axes[ax_str][range(0, nAx+1), :].T)
                        # save projections            
                        if ax_str=='catch':
                            cat_correct[i] = val
                            cat_mask[i] = mask[i]
                        elif ax_str=='target': 
                            tar_correct[i] = val
                            tar_mask[i] = mask[i]
                        elif ax_str=='pca':
                            pca_correct[i] = val
                            pca_mask[i] = mask[i]
                        elif ax_str=='delta':
                            delt_correct[i] = val
                            delt_mask[i] = mask[i]
                        elif ax_str=='tarCat':
                            tc_correct[i] = val
                            tc_mask[i] = mask[i]
                
                # compute dprime for each 
                cdp, _, _, _, _, _ = compute_dprime(cat_correct[cat_mask][np.newaxis, :], cat_correct[~cat_mask][np.newaxis, :])
                tdp, _, _, _, _, _ = compute_dprime(tar_correct[tar_mask][np.newaxis, :], tar_correct[~tar_mask][np.newaxis, :])
                pdp, _, _, _, _, _ = compute_dprime(pca_correct[pca_mask][np.newaxis, :], pca_correct[~pca_mask][np.newaxis, :])
                ddp, _, _, _, _, _ = compute_dprime(delt_correct[delt_mask][np.newaxis, :], delt_correct[~delt_mask][np.newaxis, :])
                tcdp, _, _, _, _, _ = compute_dprime(tc_correct[tc_mask][np.newaxis, :], tc_correct[~tc_mask][np.newaxis, :])

                stimulus_decoder = stimulus_decoder.append(pd.DataFrame(index=['dprime', 'axes', 'nDim', 'pair', 'snr', 'site', 'batch'],
                                                                data=[cdp, 'catch', nAx, '_'.join(pair), snr, site, batch]).T)
                stimulus_decoder = stimulus_decoder.append(pd.DataFrame(index=['dprime', 'axes', 'nDim', 'pair', 'snr', 'site', 'batch'],
                                                                data=[tdp, 'target', nAx, '_'.join(pair), snr, site, batch]).T)
                stimulus_decoder = stimulus_decoder.append(pd.DataFrame(index=['dprime', 'axes', 'nDim', 'pair', 'snr', 'site', 'batch'],
                                                                data=[pdp, 'pca', nAx, '_'.join(pair), snr, site, batch]).T)
                stimulus_decoder = stimulus_decoder.append(pd.DataFrame(index=['dprime', 'axes', 'nDim', 'pair', 'snr', 'site', 'batch'],
                                                                data=[ddp, 'delta', nAx, '_'.join(pair), snr, site, batch]).T)
                stimulus_decoder = stimulus_decoder.append(pd.DataFrame(index=['dprime', 'axes', 'nDim', 'pair', 'snr', 'site', 'batch'],
                                                                data=[tcdp, 'tarCat', nAx, '_'.join(pair), snr, site, batch]).T)


dtypes = {
    'dprime': 'float32',
    'nDim': 'float32',
    'axes': 'object',
    'pair': 'object',
    'site': 'object',
    'batch': 'object',
    'snr': 'category'
}
stimulus_decoder = stimulus_decoder.astype(dtypes)

dtypes = {
    'sound': 'object',
    'dprime': 'float32',
    'nDim': 'float32',
    'axes': 'object',
    'soundCategory': 'object',
    'site': 'object',
    'batch': 'object'
}
choice_decoder = choice_decoder.astype(dtypes)

stimulus_decoder.to_pickle(DIR + '/results/res_stimulus_decoder.pickle')
choice_decoder.to_pickle(DIR + '/results/res_choice_decoder.pickle')

## NOTE: potentiall interesting result... low SNR target decoding is *much* flatter over noise dims. Suggests that is aligns more closely with 
#       choice/noise axes (in particular for the noise axes, I think)
# sns.lineplot(x='nDim', y='dprime', hue='snr', data=stimulus_decoder[stimulus_decoder['axes'].str.contains('pca')]) 