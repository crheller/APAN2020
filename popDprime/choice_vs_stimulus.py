"""
Inspired by Ni et al 2018, look at decoding of both choice, and stimulus, 
using the noise PCs. Do noise correlations help in this case? How is performance of 
different SNRs on these axes? What is the relationship between behavior / decoding
on this axis for the different SNRs. Hypothesis is that delta correlations most
helpful for decoding low SNR targets... 

For choice, we have hit vs. miss. We also have corr reject vs. "false alarm" (incorrect hits)

Initially, just look at batch 324 / 325 for this data
"""
from settings import DIR
import popDprime.deflate_helper as dh
import popDprime.loocv_helper as lh
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy import parse_cellid
from charlieTools.ptd_ms.utils import which_rawids
import charlieTools.baphy_remote as br
import charlieTools.noise_correlations as nc
import charlieTools.preprocessing as preproc
from charlieTools.plotting import compute_ellipse
#from charlieTools.decoding import compute_dprime
from nems_lbhb.decoding import compute_dprime
from charlieTools.dim_reduction import TDR
import nems_lbhb.tin_helpers as thelp
from sklearn.decomposition import PCA
import nems.db as nd
import pickle
from itertools import combinations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

batches = [324, 325]
options = {'resp': True, 'pupil': True, 'rasterfs': 10}
recache = False
show_plots = False
start = int(0.1 * options['rasterfs'])
end = int(0.3 * options['rasterfs'])
regress_pupil = False
regress_task = False

cmap = {
    'HIT_TRIAL': 'red',
    'MISS_TRIAL': 'lightcoral',
    'CORRECT_REJECT_TRIAL': 'k',
    'INCORRECT_HIT_TRIAL': 'lightgrey',
    'CATCH': 'grey',
    'TARGET': 'red'
}

stim_decoding_results = {
    324: {-5: {'a': [], 'p': []}, 0: {'a': [], 'p': []}, np.inf: {'a': [], 'p': []}},
    325: {-5: {'a': [], 'p': []}, 0: {'a': [], 'p': []}, np.inf: {'a': [], 'p': []}}
}
choice_decoding_results = {
    324: {-5: [], 0: [], np.inf: []},
    325: {-5: [], 0: [], np.inf: []}
}
for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if (s!='CRD013b') & ('gus' not in s)]    
    for site in sites:
        rawid = None
        manager = BAPHYExperiment(batch=batch, siteid=site[:7], rawid=rawid)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()

        active_mask = ['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL', 'MISS_TRIAL']
        rec = rec.and_mask(['PASSIVE_EXPERIMENT'] + active_mask)

        rec = rec.apply_mask(reset_epochs=True)

        if regress_pupil & regress_task:
            rec = preproc.regress_state(rec, state_sigs=['pupil', 'behavior'])
        elif regress_pupil:
            rec = preproc.regress_state(rec, state_sigs=['pupil'])
        elif regress_task:
            rec = preproc.regress_state(rec, state_sigs=['behavior'])

        ra = rec.copy()
        ra = ra.create_mask(True)
        ra = ra.and_mask(active_mask)

        rp = rec.copy()
        rp = rp.create_mask(True)
        rp = rp.and_mask(['PASSIVE_EXPERIMENT'])

        # so that we only select epochs present in active (chance that
        # some passive stimuli weren't present during active)
        _rp = rp.apply_mask(reset_epochs=True)
        _ra = ra.apply_mask(reset_epochs=True)

        # =================================== find / sort epoch names ====================================
        # need to do some "hacky" stuff for batch 302 / 307 to get names to align with the TBP data
        if batch in [324, 325]:
            targets = thelp.sort_targets([f for f in _ra['resp'].epochs.name.unique() if 'TAR_' in f])
            # only keep target presented at least 5 times
            targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
            # remove "off-center targets"
            on_center = thelp.get_tar_freqs([f.strip('REM_') for f in _ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
            targets = [t for t in targets if str(on_center) in t]
            catch = [f for f in _ra['resp'].epochs.name.unique() if 'CAT_' in f]
            # remove off-center catches
            catch = [c for c in catch if str(on_center) in c]
            rem = [f for f in rec['resp'].epochs.name.unique() if 'REM_' in f]
            #targets += rem
            targets_str = targets
            catch_str = catch
            snrs = thelp.get_snrs(targets)

        # compute noise PCs using difference covariance matrix (so they're ordered by change in rsc)
        respa = []
        respp = []
        for t in targets:
            _r = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
            m = _r.mean(axis=0)
            sd = _r.std(axis=0)
            sd[sd==0] = 1
            _r = (_r - m) / sd
            respa.append(_r)

            _r = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1, keepdims=True)
            m = _r.mean(axis=0)
            sd = _r.std(axis=0)
            sd[sd==0] = 1
            _r = (_r - m) / sd
            respp.append(_r)
        respa = np.concatenate(respa, axis=0).squeeze()
        respp = np.concatenate(respp, axis=0).squeeze()

        AMAT = np.cov(respa.T); #np.fill_diagonal(AMAT, 0)
        PMAT = np.cov(respp.T); #np.fill_diagonal(PMAT, 0)
        DIFF = PMAT - AMAT

        evals, evecs = np.linalg.eig(DIFF)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        tevals = evals
        evecs = evecs[:, idx]
        tevecs = evecs

        # project data onto top two PCs, color by trial type, plot each target on different 
        # set of axes (but projecting into same space always)
        noise_axes = tevecs[:, :2]
        f, ax = plt.subplots(2, len(targets), figsize=(int(len(targets) * 4), 8), sharey=True, sharex=True)
        f2, ax2 = plt.subplots(2, len(targets), figsize=(int(len(targets) * 4), 8), sharey=True, sharex=True)
        for i, (t, snr) in enumerate(zip(targets, snrs)):
            # ACTIVE DATA
            Respc = rec['resp'].extract_epoch(catch[0], mask=ra['mask'])[:, :, start:end].mean(axis=-1)
            respc = Respc.dot(noise_axes)
            Respt = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
            respt = Respt.dot(noise_axes)

            crj_mask = rec['resp'].epoch_to_signal('CORRECT_REJECT_TRIAL').extract_epoch(catch[0], mask=ra['mask'])[:, :, start:end].mean(axis=-1).squeeze().astype(bool)
            ich_mask = rec['resp'].epoch_to_signal('INCORRECT_HIT_TRIAL').extract_epoch(catch[0], mask=ra['mask'])[:, :, start:end].mean(axis=-1).squeeze().astype(bool)
            hit_mask = rec['resp'].epoch_to_signal('HIT_TRIAL').extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1).squeeze().astype(bool)
            miss_mask = rec['resp'].epoch_to_signal('MISS_TRIAL').extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1).squeeze().astype(bool)
            if crj_mask.sum()>2:
                ax[0, i].scatter(respc[crj_mask, 0], respc[crj_mask, 1], color=cmap['CORRECT_REJECT_TRIAL'])
                el = compute_ellipse(respc[crj_mask, 0], respc[crj_mask, 1])
                ax[0, i].plot(el[0], el[1], color=cmap['CORRECT_REJECT_TRIAL'], lw=2, label='C.R.')
            if ich_mask.sum()>2:
                ax[0, i].scatter(respc[ich_mask, 0], respc[ich_mask, 1], color=cmap['INCORRECT_HIT_TRIAL'])
                el = compute_ellipse(respc[ich_mask, 0], respc[ich_mask, 1])
                ax[0, i].plot(el[0], el[1], color=cmap['INCORRECT_HIT_TRIAL'], lw=2, label='F.A.')
            if hit_mask.sum()>2:
                ax[0, i].scatter(respt[hit_mask, 0], respt[hit_mask, 1], color=cmap['HIT_TRIAL'])
                el = compute_ellipse(respt[hit_mask, 0], respt[hit_mask, 1])
                ax[0, i].plot(el[0], el[1], color=cmap['HIT_TRIAL'], lw=2, label='HIT')
            if miss_mask.sum()>2:
                ax[0, i].scatter(respt[miss_mask, 0], respt[miss_mask, 1], color=cmap['MISS_TRIAL'])
                el = compute_ellipse(respt[miss_mask, 0], respt[miss_mask, 1])
                ax[0, i].plot(el[0], el[1], color=cmap['MISS_TRIAL'], lw=2, label='MISS')

            catchel = compute_ellipse(respc[:, 0], respc[:, 1])
            tarel = compute_ellipse(respt[:, 0], respt[:, 1])
            ax[0, i].plot(catchel[0], catchel[1], color=cmap['CATCH'], lw=2, linestyle='--')
            ax[0, i].plot(tarel[0], tarel[1], color=cmap['TARGET'], lw=2, linestyle='--')
            ax[0, i].set_title(t)
            ax[0, i].set_ylabel(r'Noise $PC_2$')
            ax[0, i].set_xlabel(r'Noise $PC_1$')
            ax[0, i].legend(frameon=False)

            # ===== perform leave-one-out decoding analysis on active data =====

            # choice decoding (hit vs. miss) if there are enough trials
            prop_correct = []
            if (hit_mask.sum() > 2) & (miss_mask.sum() > 2):
                j = 0
                while (j < (hit_mask.sum()-1)) & (j < (miss_mask.sum()-1)):
                    respt = Respt.dot(tevecs[:, :(j+1)])
                    A = respt[hit_mask, :] 
                    B = respt[miss_mask, :]
                    prop_correct.append(lh.get_proportion_correct(A, B))
                    j += 1
            ax2[0, i].plot(prop_correct, 'o-', label='Choice decoding')
            choice_decoding_results[batch][snr].append(prop_correct)

            # stimulus decoding
            prop_correct = []
            j = 0
            while (j < (Respt.shape[0]-1)) & (j < (Respc.shape[0]-1)):
                respt = Respt.dot(tevecs[:, :(j+1)])
                respc = Respc.dot(tevecs[:, :(j+1)])
                A = respt
                B = respc
                prop_correct.append(lh.get_proportion_correct(A, B))
                j += 1
            stim_decoding_results[batch][snr]['a'].append(prop_correct)

            ax2[0, i].plot(prop_correct, 'o-', label='Stimulus decoding')
            ax2[0, i].set_xlabel("Number of PCs")
            ax2[0, i].set_ylabel('Proportion correct')
            ax2[0, i].legend(frameon=False)
            ax2[0, i].set_title(t)

            # PASSIVE DATA
            Respc = rec['resp'].extract_epoch(catch[0], mask=rp['mask'])[:, :, start:end].mean(axis=-1)
            respc = Respc.dot(noise_axes)
            Respt = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
            respt = Respt.dot(noise_axes)

            ax[1, i].scatter(respc[:, 0], respc[:, 1], color=cmap['CORRECT_REJECT_TRIAL'])
            ax[1, i].scatter(respt[:, 0], respt[:, 1], color=cmap['HIT_TRIAL'])
            catchel = compute_ellipse(respc[:, 0], respc[:, 1])
            tarel = compute_ellipse(respt[:, 0], respt[:, 1])
            ax[1, i].plot(catchel[0], catchel[1], color=cmap['CATCH'], lw=3)
            ax[1, i].plot(tarel[0], tarel[1], color=cmap['TARGET'], lw=3)
            ax[1, i].set_ylabel(r'Noise $PC_2$')
            ax[1, i].set_xlabel(r'Noise $PC_1$')

            # ===== perform leave-one-out decoding analysis on passive data =====
            # stimulus decoding
            prop_correct = []
            j = 0
            while (j < (Respt.shape[0]-1)) & (j < (Respc.shape[0]-1)):
                respt = Respt.dot(tevecs[:, :(j+1)])
                respc = Respc.dot(tevecs[:, :(j+1)])
                A = respt
                B = respc
                prop_correct.append(lh.get_proportion_correct(A, B))
                j += 1
            stim_decoding_results[batch][snr]['p'].append(prop_correct)
            ax2[1, i].plot(prop_correct, 'o-', label='Stimulus decoding', color='tab:orange')
            ax2[1, i].set_xlabel("Number of PCs")
            ax2[1, i].set_ylabel('Proportion correct')
            ax2[1, i].legend(frameon=False)

        f.tight_layout()
        f2.tight_layout()
        
if not show_plots:
    plt.close('all')

# summary per area of active decoding results vs. PC
nPCs = 8
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# A1 stim decoding
for snr in stim_decoding_results[324].keys():
    resa = stim_decoding_results[324][snr]['a']
    resa = np.stack([r[:nPCs] for r in resa])
    ax[0].errorbar(range(0, nPCs), resa.mean(axis=0), yerr=resa.std(axis=0)/np.sqrt(resa.shape[0]), capsize=3, marker='o', label=f'Active {snr}dB')

    resa = stim_decoding_results[325][snr]['a']
    resa = np.stack([r[:nPCs] for r in resa])
    ax[1].errorbar(range(0, nPCs), resa.mean(axis=0), yerr=resa.std(axis=0)/np.sqrt(resa.shape[0]), capsize=3, marker='o', label=f'Active {snr}dB')

ax[0].legend(frameon=False)
ax[0].set_xlabel('PC')
ax[0].set_ylabel('Proportion Correct')
ax[0].set_title('A1')

ax[1].legend(frameon=False)
ax[1].set_xlabel('PC')
ax[1].set_ylabel('Proportion Correct')
ax[1].set_title('PEG')

f.tight_layout()


# summary per area of delta decoding results vs. PC
nPCs = 8
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# A1 stim decoding
for snr in stim_decoding_results[324].keys():
    resa = stim_decoding_results[324][snr]['a']
    resa = np.stack([r[:nPCs] for r in resa])
    resp = stim_decoding_results[324][snr]['p']
    resp = np.stack([r[:nPCs] for r in resp])
    diff = resa - resp

    ax[0].errorbar(range(0, nPCs), diff.mean(axis=0), yerr=diff.std(axis=0)/np.sqrt(diff.shape[0]), capsize=3, marker='o', label=f'{snr}dB')

    resa = stim_decoding_results[325][snr]['a']
    resa = np.stack([r[:nPCs] for r in resa])
    resp = stim_decoding_results[325][snr]['p']
    resp = np.stack([r[:nPCs] for r in resp])
    diff = resa - resp

    ax[1].errorbar(range(0, nPCs), diff.mean(axis=0), yerr=diff.std(axis=0)/np.sqrt(diff.shape[0]), capsize=3, marker='o', label=f'{snr}dB')

ax[0].legend(frameon=False)
ax[0].set_xlabel('PC')
ax[0].set_ylabel('Change in Proportion Correct\n(Act. - Pas.)')
ax[0].set_title('A1')

ax[1].legend(frameon=False)
ax[1].set_xlabel('PC')
ax[1].set_ylabel('Change in Proportion Correct\n(Act. - Pas.)')
ax[1].set_title('PEG')

f.tight_layout()



plt.show()