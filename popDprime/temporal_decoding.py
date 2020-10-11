"""
Sliding window with 50 ms (?) bins. So time course of decoding
over trial. Does optimal decoding axis change over the course of the trial?
How do the different SNR targets line up?

CORRECT TRIALS ONLY

Fig1 (single decoding axis computed during decision window):
2 panels:
    active / passive "pop psth" on dU

Fig2 (new decoding axis for each time point):
8 panels: 
    active / passive "pop psth" on dU axis
    active / passive sim matrix for dU over time bins
    active / passive sim matrix for wopt over time bins (says some about how noise is changing)
    active / passive sim matrix for noise axis 1 over time bins
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.baphy_remote as br
import charlieTools.noise_correlations as nc
import charlieTools.preprocessing as preproc
from charlieTools.plotting import compute_ellipse
from charlieTools.decoding import compute_dprime
from charlieTools.dim_reduction import TDR
import nems_lbhb.tin_helpers as thelp
from sklearn.decomposition import PCA
import nems.db as nd
from itertools import combinations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#empl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['font.size'] = 14

# fig path
fpath = '/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/figures/TemporalDecoding/'
res_path = '/home/charlie/Desktop/lbhb/code/APAN2020/results/'
savefig = True

# recording load options
options = {'resp': True, 'pupil': True, 'rasterfs': 100}
batches = [324, 325]
recache = False

nTDRAx = 3       # first, perform dimensionality reduction (for stability of decoding algorithm)
regress_pupil = False  # regress out first order pupil?

# define decision (pre-lick) window
start = int(0.1 * options['rasterfs'])
end = int(0.3 * options['rasterfs'])
tstarts = np.arange(0, 0.46, 0.01) # 10 ms steps, 50 ms bins
tends = np.arange(0.05, 0.51, 0.01)
sidx = [int(t * options['rasterfs']) for t in tstarts]
eidx = [int(t * options['rasterfs']) for t in tends]
for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if s!='CRD013b']
    for site in sites:
        manager = BAPHYExperiment(batch=batch, siteid=site)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()

        # regress out first order pupil
        if regress_pupil:
            rec = preproc.regress_state(rec, state_sigs=['pupil'])

        ra = rec.copy()
        ra = ra.create_mask(True)
        ra = ra.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])

        rp = rec.copy()
        rp = rp.create_mask(True)
        rp = rp.and_mask(['PASSIVE_EXPERIMENT'])

        # find / sort epoch names
        targets = thelp.sort_targets([f for f in ra['resp'].epochs.name.unique() if 'TAR_' in f])
        targets = [t for t in targets if (ra['resp'].epochs.name==t).sum()>=5]
        on_center = thelp.get_tar_freqs([f.strip('REM_') for f in ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
        targets = [t for t in targets if str(on_center) in t]
        catch = [f for f in ra['resp'].epochs.name.unique() if 'CAT_' in f]
        catch = [c for c in catch if str(on_center) in c]
        ref_stim = thelp.sort_refs([f for f in ra['resp'].epochs.name.unique() if 'STIM_' in f])

        BwG, gR = thelp.make_tbp_colormaps(ref_stim, targets+catch)

        # get resp dictionaries
        da = ra['resp'].extract_epochs(targets + catch, mask=ra['mask'])
        dp = ra['resp'].extract_epochs(targets + catch, mask=rp['mask'])

        # ====== perform analysis for each time bin in each behavior state ======
        time = []
        projections_a = dict.fromkeys(targets + catch)
        projections_p = dict.fromkeys(targets + catch)
        dUa = []
        dUp = []
        eveca = []
        evecp = []
        for i, (s, e) in enumerate(zip(sidx, eidx)):
            time.append(np.mean([s, e]) / options['rasterfs'])


            # get TDR dims first to reduced matrix dim for decoding
            talla = np.concatenate([r[:, :, s:e] for k, r in da.items() if ('TAR_' in k)], axis=0)
            calla = np.concatenate([r[:, :, s:e] for k, r in da.items() if ('CAT_' in k)], axis=0)
            tallp = np.concatenate([r[:, :, s:e] for k, r in dp.items() if ('TAR_' in k)], axis=0)
            callp = np.concatenate([r[:, :, s:e] for k, r in dp.items() if ('CAT_' in k)], axis=0)
            tall = np.concatenate([talla, tallp], axis=0)
            call = np.concatenate([calla, callp], axis=0)

            tdr = TDR(n_additional_axes=nTDRAx-1)
            tdr.fit(tall.mean(axis=-1), call.mean(axis=-1))

            # ==== perform decoding projections ====
            # active data
            tall = np.concatenate([r[:, :, s:e] for k, r in da.items() if ('TAR_' in k)], axis=0).mean(axis=-1).dot(tdr.weights.T)
            call = np.concatenate([r[:, :, s:e] for k, r in da.items() if ('CAT_' in k)], axis=0).mean(axis=-1).dot(tdr.weights.T)
            dprime, wopt, evals, evecs, _, dU = compute_dprime(tall.T, call.T)
            dU = dU / np.linalg.norm(dU)
            dUa.append(dU)
            eveca.append(evecs[:, 0])

            for t in targets + catch:
                r = da[t][:, :, s:e].mean(axis=-1).dot(tdr.weights.T).dot(dU.T)
                if i == 0:
                    projections_a[t] = r
                else:
                    projections_a[t] = np.concatenate([projections_a[t], r], axis=-1)

            # passive data
            tall = np.concatenate([r[:, :, s:e] for k, r in dp.items() if ('TAR_' in k)], axis=0).mean(axis=-1).dot(tdr.weights.T)
            call = np.concatenate([r[:, :, s:e] for k, r in dp.items() if ('CAT_' in k)], axis=0).mean(axis=-1).dot(tdr.weights.T)
            dprime, wopt, evals, evecs, _, dU = compute_dprime(tall.T, call.T)
            dU = dU / np.linalg.norm(dU)
            dUp.append(dU)
            evecp.append(evecs[:, 0])

            for t in targets + catch:
                r = dp[t][:, :, s:e].mean(axis=-1).dot(tdr.weights.T).dot(dU.T)
                if i == 0:
                    projections_p[t] = r
                else:
                    projections_p[t] = np.concatenate([projections_p[t], r], axis=-1)

        # ===== PLOT RESULTS =====
        
        f = plt.figure(figsize=(12, 6))
        
        psth1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
        psth2 = plt.subplot2grid((2, 4), (1, 0), colspan=2, sharey=psth1)
        sim1 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
        sim2 = plt.subplot2grid((2, 4), (1, 2), colspan=1)
        nsim1 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
        nsim2 = plt.subplot2grid((2, 4), (1, 3), colspan=1)

        psth1.set_title('Active')
        for i, t in enumerate(catch+targets):
            m = projections_a[t].mean(axis=0)
            sem = projections_a[t].std(axis=0) #/ np.sqrt(projections_a[t].shape[0])
            psth1.fill_between(time, m-sem, m+sem, lw=0, color=gR(i), alpha=0.3)
            psth1.plot(time, m, color=gR(i), lw=2, label=t)

        psth2.set_title('Passive')
        for i, t in enumerate(catch+targets):
            m = projections_p[t].mean(axis=0)
            sem = projections_p[t].std(axis=0) #/ np.sqrt(projections_p[t].shape[0])
            psth2.fill_between(time, m-sem, m+sem, lw=0, color=gR(i), alpha=0.3)
            psth2.plot(time, m, color=gR(i), lw=2, label=t)

        psth1.axvline(0.1, linestyle='--', color='k')
        psth1.axvline(0.4, linestyle='--', color='k')
        psth1.axhline(0, linestyle='--', color='grey')
        psth2.axvline(0.1, linestyle='--', color='k')
        psth2.axvline(0.4, linestyle='--', color='k')
        psth2.axhline(0, linestyle='--', color='grey')
        
        psth1.legend(frameon=False)
        psth2.set_xlabel('Time(s)')
        psth1.set_ylabel(r"$\Delta \mathbf{\mu}$ Projection")
        psth2.set_ylabel(r"$\Delta \mathbf{\mu}$ Projection")

        dUa = np.stack(dUa).squeeze()
        duSima = dUa.dot(dUa.T)
        pcm = sim1.imshow(duSima, aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=[time[0], time[-1], time[0], time[-1]])
        f.colorbar(pcm, ax=sim1)

        dUp = np.stack(dUp).squeeze()
        duSimp = dUp.dot(dUp.T)
        pcm = sim2.imshow(duSimp, aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=[time[0], time[-1], time[0], time[-1]])
        f.colorbar(pcm, ax=sim2)

        eveca = abs(np.stack(eveca).squeeze())
        evecSima = eveca.dot(eveca.T)
        pcm = nsim1.imshow(evecSima, aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=[time[0], time[-1], time[0], time[-1]])
        f.colorbar(pcm, ax=nsim1)

        evecp = abs(np.stack(evecp).squeeze())
        evecSimp = eveca.dot(evecp.T)
        pcm = nsim2.imshow(evecSimp, aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=[time[0], time[-1], time[0], time[-1]])
        f.colorbar(pcm, ax=nsim2)

        for s in [sim1, sim2]:
            s.set_xlabel('Time (s)')
            s.set_ylabel('Time (s)')
            s.set_title(r"$\Delta \mathbf{\mu}$ Similarity")
            s.axvline(0.1, linestyle='--', color='k')
            s.axhline(0.1, linestyle='--', color='k')
            s.axvline(0.4, linestyle='--', color='k')
            s.axhline(0.4, linestyle='--', color='k')
        for s in [nsim1, nsim2]:
            s.set_xlabel('Time (s)')
            s.set_ylabel('Time (s)')
            s.set_title(r"Noise ($\mathbf{e}_1$) Similarity")
            s.axvline(0.1, linestyle='--', color='k')
            s.axhline(0.1, linestyle='--', color='k')
            s.axvline(0.4, linestyle='--', color='k')
            s.axhline(0.4, linestyle='--', color='k')


        f.tight_layout()

        if savefig:
            f.savefig(fpath + f"TemporalDecoding_{site}.pdf")
        


        # ======= Same analysis, but fixed decoding axis ==============
plt.show()

