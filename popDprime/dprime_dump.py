"""
Quick and dirty quantify discriminability of pairwise combos of tars / catches.
In active / passive.
    deal with pupil?

define TDR over all stims? Or on pairwise basis? Both? Use PC-space too?
"""
from settings import DIR
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy import parse_cellid
from charlieTools.ptd_ms.utils import which_rawids
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
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 14

# fig path
fpath = DIR+ 'results/figures/EllipsePlots/'
res_path = DIR + 'results/'

# recording load options
batches = [302, 307, 324, 325]
Aoptions = dict.fromkeys(batches)
Aoptions[302] = {'resp': True, 'pupil': True, 'rasterfs': 10}
Aoptions[307] = {'resp': True, 'pupil': True, 'rasterfs': 20}
Aoptions[324] = {'resp': True, 'pupil': True, 'rasterfs': 10}
Aoptions[325] = {'resp': True, 'pupil': True, 'rasterfs': 10}
recache = False

# state-space projection options
zscore = False

# regress out first order pupil?
regress_pupil = False
regress_task = False

# plot ref
plot_ref = False
if plot_ref:
    fext = '_withREF'
else:
    fext = ''

# extract evoked periods before lick only
dec_window = {
    302: {'start': int(0.1 * Aoptions[302]['rasterfs']), 'end': int(0.3 * Aoptions[302]['rasterfs'])},
    307: {'start': int(0.35 * Aoptions[307]['rasterfs']), 'end': int(0.55 * Aoptions[307]['rasterfs'])},
    324: {'start': int(0.1 * Aoptions[324]['rasterfs']), 'end': int(0.3 * Aoptions[324]['rasterfs'])},
    325: {'start': int(0.1 * Aoptions[325]['rasterfs']), 'end': int(0.3 * Aoptions[325]['rasterfs'])},
    }

# siteids
dfs = []
for batch in batches:
    start = dec_window[batch]['start']
    end = dec_window[batch]['end']
    options = Aoptions[batch]
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if (s!='CRD013b') & ('gus' not in s)]    
    if batch == 302:
        sites1 = [s+'.e1:64' for s in sites]
        sites2 = [s+'.e65:128' for s in sites]
        sites = sites1 + sites2
    for site in sites:
        skip_site = False
        # set up subplots for PCA / TDR projections
        f, ax = plt.subplots(2, 2, figsize=(12, 10))
        f.canvas.set_window_title(site)
            
        if batch == 307:
            rawid = which_rawids(site)
        else:
            rawid = None
        print("Analyzing site: {}".format(site))
        manager = BAPHYExperiment(batch=batch, siteid=site[:7], rawid=rawid)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()
        if batch == 302:
            c, _ = parse_cellid({'cellid': site, 'batch': batch})
            rec['resp'] = rec['resp'].extract_channels(c)

        # mask appropriate trials
        if batch in [324, 325]:
            active_mask = ['HIT_TRIAL', 'CORRECT_REJECT_TRIAL']
            rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
        elif batch == 307:
            active_mask = ['HIT_TRIAL']
            rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL'])
        elif batch == 302:
            active_mask = ['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL']
            rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL'])
        
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

        _rp = rp.apply_mask(reset_epochs=True)
        _ra = ra.apply_mask(reset_epochs=True)

        # =================================== find / sort epoch names ====================================
        # need to do some "hacky" stuff for batch 302 / 307 to get names to align with the TIN data
        if batch in [324, 325]:
            targets = thelp.sort_targets([f for f in _ra['resp'].epochs.name.unique() if 'TAR_' in f])
            # only keep target presented at least 5 times
            targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
            # remove "off-center targets"
            on_center = thelp.get_tar_freqs([f.strip('REM_') for f in _ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
            targets = [t for t in targets if str(on_center) in t]
            if len(targets)==0:
                # NOT ENOUGH REPS AT THIS SITE
                skip_site = True
            catch = [f for f in _ra['resp'].epochs.name.unique() if 'CAT_' in f]
            # remove off-center catches
            catch = [c for c in catch if str(on_center) in c]
            rem = [f for f in rec['resp'].epochs.name.unique() if 'REM_' in f]
            targets_str = targets
            catch_str = catch
            ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
            ref_str = ref_stim
            tar_idx = 0
        elif batch == 307:
            params = manager.get_baphy_exptparams()
            params = [p for p in params if p['BehaveObjectClass']!='Passive'][0]
            tf = params['TrialObject'][1]['TargetHandle'][1]['Names']
            targets = [f'TAR_{t}' for t in tf]
            if params['TrialObject'][1]['OverlapRefTar']=='Yes':
                snrs = params['TrialObject'][1]['RelativeTarRefdB'] 
            else:
                snrs = ['Inf']
            snrs = [s if (s!=np.inf) else 'Inf' for s in snrs]
            #catchidx = int(params['TrialObject'][1]['OverlapRefIdx'])
            refs = params['TrialObject'][1]['ReferenceHandle'][1]['Names']
            catch = ['REFERENCE'] #['STIM_'+refs[catchidx]]
            catch_str = [f'CAT_{tf[0]}+-InfdB+Noise+allREFs']
            targets_str = [f'TAR_{t}+{snr}dB+Noise' for snr, t in zip(snrs, tf)]
            targets_str = targets_str[::-1]
            targets = targets[::-1]

            # only keep targets w/ at least 5 reps in active
            targets_str = [ts for t, ts in zip(targets, targets_str) if (_ra['resp'].epochs.name==t).sum()>=5]
            targets = [t for t in targets if (_ra['resp'].epochs.name==t).sum()>=5]
            
            ref_stim = [f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f]
            ref_str = [f"STIM_{tf[0]}+torc{r.split('LIN_')[1].split('_v')[0]}" for r in ref_stim]

            # only keep refs with at least 3 reps
            ref_str = [ts for t, ts in zip(ref_stim, ref_str) if (_ra['resp'].epochs.name==t).sum()>=3]
            ref_stim = [t for t in ref_stim if (_ra['resp'].epochs.name==t).sum()>=3]

            tar_idx = 0

        elif batch == 302:
            params = manager.get_baphy_exptparams()
            params = [p for p in params if p['BehaveObjectClass']!='Passive'][0]
            tf = params['TrialObject'][1]['TargetHandle'][1]['Names']
            targets = [f'TAR_{t}' for t in tf]
            pdur = params['BehaveObject'][1]['PumpDuration']
            rew = np.array(tf)[np.array(pdur)==1].tolist()
            catch = [t for t in targets if (t.split('TAR_')[1] not in rew)]
            catch_str = [(t+'+InfdB+Noise').replace('TAR_', 'CAT_') for t in targets if (t.split('TAR_')[1] not in rew)]
            targets = [t for t in targets if (t.split('TAR_')[1] in rew)]
            targets_str = [t+'+InfdB+Noise' for t in targets if (t.split('TAR_')[1] in rew)]
            ref_stim = thelp.sort_refs([f for f in _ra['resp'].epochs.name.unique() if 'STIM_' in f])
            ref_str = ref_stim
            tar_idx = 1

        sounds = targets + catch

        if not skip_site:
            # define colormaps for each sound
            BwG, gR = thelp.make_tbp_colormaps(ref_str, catch_str+targets_str, use_tar_freq_idx=tar_idx)
            # get all pairwise combos of targets / catches
            pairs = list(combinations(['REFERENCE', 'TARGET'] + ref_stim + sounds, 2))
            pairs_str = list(combinations(['REFERENCE', 'TARGET'] + ref_str + targets_str + catch_str, 2))
            pairs_str = [p for p, p_ in zip(pairs_str, pairs) if p_[0]!=p_[1]]
            pairs = [p for p in pairs if p[0]!=p[1]]
            if len(targets)==1:
                pairs = [p for p in pairs if (p!=('TARGET', targets[0])) & (p!=(targets[0], 'TARGET'))]
                pairs_str = [p for p in pairs_str if (p!=('TARGET', targets[0])) & (p!=(targets[0], 'TARGET'))]

            df = pd.DataFrame() 

            # get overall TDR axes (grouping target / catch)
            tar = np.vstack([v[:, :, start:end].mean(axis=-1) for (k, v) in rec['resp'].extract_epochs(targets, mask=rec['mask']).items()])
            cat = np.vstack([v[:, :, start:end].mean(axis=-1) for (k, v) in rec['resp'].extract_epochs(catch, mask=rec['mask']).items()])
            m = np.concatenate((tar, cat), axis=0).mean(axis=0)
            sd = np.concatenate((tar, cat), axis=0).std(axis=0)
            sd[sd==0] = 1
            if not zscore:
                m = 0
                sd = 1
            tar = (tar - m) / sd
            cat = (cat - m) / sd
            tdr = TDR()
            tdr.fit(tar, cat)
            all_tdr_weights = tdr.weights

            # get first two PCs of REF space, and try decoding there
            dref = rec['resp'].extract_epochs(ref_stim, mask=rec['mask'])
            mpca = np.concatenate([dref[e][:, :, start:end] for e in dref.keys()], axis=0).mean(axis=-1).mean(axis=0)
            sdpc = np.concatenate([dref[e][:, :, start:end] for e in dref.keys()], axis=0).mean(axis=-1).std(axis=0)
            sdpc[sdpc==0] = 1
            if not zscore:
                mpca = 0
                sdpc = 1

            dref = {k: (v.transpose(0, -1, 1) - mpca).transpose(0, -1, 1)  for (k, v) in dref.items()}
            dref = {k: (v.transpose(0, -1, 1) / sdpc).transpose(0, -1, 1)  for (k, v) in dref.items()}
            Rall_u = np.vstack([dref[k].sum(axis=2).mean(axis=0) for k in dref.keys()])
            pca = PCA(n_components=2)
            pca.fit(Rall_u)
            pc_axes = pca.components_

            # plot projections for data into the PCA space and "all" TDR space
            # REF
            if plot_ref:
                for i, t in enumerate(ref_stim):
                    # ================================ TDR ==========================================
                    r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(all_tdr_weights.T).T
                    ax[0, 0].set_title('Active')
                    ax[0, 0].scatter(r1[0], r1[1], alpha=0.8, s=15, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[0, 0].plot(el[0], el[1], color=BwG(i), alpha=0.8, label=ref_str[i].split('STIM_')[1], lw=2)

                    r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(all_tdr_weights.T).T
                    ax[0, 1].set_title('Passive')
                    ax[0, 1].scatter(r1[0], r1[1], alpha=0.8, s=15, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[0, 1].plot(el[0], el[1], color=BwG(i), alpha=0.8, lw=2)

                    # =============================== PCA ========================================
                    r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(pc_axes.T).T
                    ax[1, 0].set_title('Active')
                    ax[1, 0].scatter(r1[0], r1[1], alpha=0.8, s=15, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[1, 0].plot(el[0], el[1], color=BwG(i), alpha=0.8, lw=2)

                    r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                    r1 = (r1 - m) / sd
                    r1 = r1.dot(pc_axes.T).T
                    ax[1, 1].set_title('Passive')
                    ax[1, 1].scatter(r1[0], r1[1], alpha=0.8, s=15, lw=0, color=BwG(i))
                    el = thelp.compute_ellipse(r1[0], r1[1])
                    ax[1, 1].plot(el[0], el[1], color=BwG(i), alpha=0.8, lw=2)
                

            # TARGETS / CATCHES
            for i, (t, ts) in enumerate(zip(catch + targets, catch_str+targets_str)):
                # ================================ TDR ==========================================
                r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(all_tdr_weights.T).T
                ax[0, 0].set_title('Active')
                ax[0, 0].scatter(r1[0], r1[1], alpha=1, s=15, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[0, 0].plot(el[0], el[1], color=gR(i), label=ts, lw=2)

                r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(all_tdr_weights.T).T
                ax[0, 1].set_title('Passive')
                ax[0, 1].scatter(r1[0], r1[1], alpha=1, s=15, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[0, 1].plot(el[0], el[1], color=gR(i), lw=2)

                # =============================== PCA ========================================
                r1 = rec['resp'].extract_epoch(t, mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(pc_axes.T).T
                ax[1, 0].set_title('Active')
                ax[1, 0].scatter(r1[0], r1[1], alpha=1, s=15, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[1, 0].plot(el[0], el[1], color=gR(i), label=ts, lw=2)

                r1 = rec['resp'].extract_epoch(t, mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r1 = r1.dot(pc_axes.T).T
                ax[1, 1].set_title('Passive')
                ax[1, 1].scatter(r1[0], r1[1], alpha=1, s=15, lw=0, color=gR(i))
                el = thelp.compute_ellipse(r1[0], r1[1])
                ax[1, 1].plot(el[0], el[1], color=gR(i), lw=2)


            ylims = (np.min([ax[0, 0].get_ylim()[0], ax[0, 1].get_ylim()[0]]), np.max([ax[0, 0].get_ylim()[1], ax[0, 1].get_ylim()[1]]))
            xlims = (np.min([ax[0, 0].get_xlim()[0], ax[0, 1].get_xlim()[0]]), np.max([ax[0, 0].get_xlim()[1], ax[0, 1].get_xlim()[1]]))
            ax[0, 0].set_xlim(xlims)
            ax[0, 0].set_ylim(ylims)
            ax[0, 1].set_xlim(xlims)
            ax[0, 1].set_ylim(ylims)

            ylims = (np.min([ax[1, 0].get_ylim()[0], ax[1, 1].get_ylim()[0]]), np.max([ax[1, 0].get_ylim()[1], ax[1, 1].get_ylim()[1]]))
            xlims = (np.min([ax[1, 0].get_xlim()[0], ax[1, 1].get_xlim()[0]]), np.max([ax[1, 0].get_xlim()[1], ax[1, 1].get_xlim()[1]]))
            ax[1, 0].set_xlim(xlims)
            ax[1, 0].set_ylim(ylims)
            ax[1, 1].set_xlim(xlims)
            ax[1, 1].set_ylim(ylims)

            if batch==307:
                leg = ax[0, 0].legend(frameon=False, handlelength=0, bbox_to_anchor=(-0.05, 1.0), loc='upper right', fontsize=6)
            else:
                leg = ax[0, 0].legend(frameon=False, handlelength=0, bbox_to_anchor=(-0.05, 1.0), loc='upper right')
            for line, text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
            ax[0, 0].set_xlabel(r"$TDR_1$ ($\Delta \mu$)")
            ax[0, 1].set_xlabel(r"$TDR_1$ ($\Delta \mu$)")
            ax[0, 0].set_ylabel(r"$TDR_2$")
            ax[0, 1].set_ylabel(r"$TDR_2$")

            ax[1, 0].set_xlabel(r"$PC_1$")
            ax[1, 1].set_xlabel(r"$PC_1$")
            ax[1, 0].set_ylabel(r"$PC_2$")
            ax[1, 1].set_ylabel(r"$PC_2$")

            f.tight_layout()

            if zscore:
                if regress_pupil & regress_task:
                    f.savefig(fpath + f'{site}{fext}_zscore_pr_br.pdf')
                elif regress_task:
                    f.savefig(fpath + f'{site}{fext}_zscore_br.pdf')
                elif regress_pupil:
                    f.savefig(fpath + f'{site}{fext}_zscore_pr.pdf')
                else:
                    f.savefig(fpath + f'{site}{fext}_zscore.pdf')
            else:
                if regress_pupil & regress_task:
                    f.savefig(fpath + f'{site}{fext}_pr_br.pdf')
                elif regress_task:
                    f.savefig(fpath + f'{site}{fext}_br.pdf')
                elif regress_pupil:
                    f.savefig(fpath + f'{site}{fext}_pr.pdf')
                else:
                    f.savefig(fpath + f'{site}{fext}.pdf')

            # get behavior performance for this site
            behavior_performance = manager.get_behavior_performance(**options)

            # for each pair, project into TDR (overall and pair-specific) and compute dprime
            for i, (pair, pair_str) in enumerate(zip(pairs, pairs_str)):
                print(f"pair {i}/{len(pairs)}")
                idx = pair[0] + '_' + pair[1]

                if ('STIM_' in pair_str[0]) | ('REFERENCE' in pair_str[0]) | ('TARGET' in pair_str[0]): snr1 = np.inf
                else: snr1 = thelp.get_snrs([pair_str[0]])[0]
                if ('STIM_' in pair_str[1]) | ('REFERENCE' in pair_str[1]) | ('TARGET' in pair_str[1]): snr2 = np.inf
                else: snr2 = thelp.get_snrs([pair_str[1]])[0]
                
                if ('REFERENCE' in pair_str[0]) | ('TARGET' in pair_str[0]): f1 = 0
                else: f1 = thelp.get_tar_freqs([pair_str[0].strip('STIM_')])[0]
                if ('REFERENCE' in pair_str[1]) | ('TARGET' in pair_str[1]): f2 = 0
                else: f2 = thelp.get_tar_freqs([pair_str[1].strip('STIM_')])[0]

                cat_cat = ('CAT_' in pair_str[0]) & ('CAT_' in pair_str[1])
                tar_tar = ('TAR_' in pair_str[0]) & ('TAR_' in pair_str[1])
                cat_tar = (('CAT_' in pair_str[0]) & ('TAR_' in pair_str[1])) | (('CAT_' in pair_str[1]) & ('TAR_' in pair_str[0]))
                ref_tar = (('STIM_' in pair_str[0]) & ('TAR_' in pair_str[1])) | (('STIM_' in pair_str[1]) & ('TAR_' in pair_str[0]))
                ref_ref = ('STIM_' in pair_str[0]) & ('STIM_' in pair_str[1])
                ref_cat = (('STIM_' in pair_str[0]) & ('CAT_' in pair_str[1])) | (('STIM_' in pair_str[1]) & ('CAT_' in pair_str[0]))
                aref_tar = (('REFERENCE' in pair_str[0]) & ('TAR_' in pair_str[1])) | (('REFERENCE' in pair_str[1]) & ('TAR_' in pair_str[0]))
                aref_cat = (('REFERENCE' in pair_str[0]) & ('CAT_' in pair_str[1])) | (('REFERENCE' in pair_str[1]) & ('CAT_' in pair_str[0]))
                aref_ref = (('REFERENCE' in pair_str[0]) & ('STIM_' in pair_str[1])) | (('REFERENCE' in pair_str[1]) & ('STIM_' in pair_str[0]))
                atar_ref = (('TARGET' in pair_str[0]) & ('STIM_' in pair_str[1])) | (('TARGET' in pair_str[1]) & ('STIM_' in pair_str[0]))
                atar_aref = (('TARGET' in pair_str[0]) & ('REFERENCE' in pair_str[1])) | (('TARGET' in pair_str[1]) & ('REFERENCE' in pair_str[0]))
                atar_cat = (('TARGET' in pair_str[0]) & ('CAT_' in pair_str[1])) | (('TARGET' in pair_str[1]) & ('CAT_' in pair_str[0]))

                if sum([cat_cat, tar_tar, cat_tar, ref_tar, ref_ref, ref_cat, aref_tar, aref_cat, aref_ref, atar_ref, atar_aref, atar_cat]) > 1:
                    raise ValueError("Ambiguous stimulus pair")

                # get behavioral DI
                if ('REFERENCE' in pair_str[0]) & (('TAR_' in pair_str[1]) | ('CAT_' in pair_str[1])):
                    di = behavior_performance['DI'][pair[1].strip('TAR_').strip('CAT_')]
                elif ('REFERENCE' in pair_str[1]) & (('TAR_' in pair_str[0]) | ('CAT_' in pair_str[0])):
                    di = behavior_performance['DI'][pair[0].strip('TAR_').strip('CAT_')]
                elif ('STIM_' in pair_str[0]) | ('STIM_' in pair_str[1]):
                    di = np.inf
                elif ('TAR_' in pair_str[0]) & ('CAT_' in pair_str[1]):
                    try:
                        di = behavior_performance['LI'][pair[0].strip('TAR_').strip('CAT_')+'_'+pair[1].strip('TAR_').strip('CAT_')]
                    except:
                        di = behavior_performance['DI'][pair[0].strip('TAR_').strip('CAT_')]
                elif ('TAR_' in pair_str[1]) & ('CAT_' in pair_str[0]):
                    try:
                        di = behavior_performance['LI'][pair[1].strip('TAR_').strip('CAT_')+'_'+pair[0].strip('TAR_').strip('CAT_')]
                    except:
                        di = behavior_performance['DI'][pair[1].strip('TAR_').strip('CAT_')]
                else:
                    di = np.inf
                # extract data over all trials for TDR
                r1 = rec['resp'].extract_epoch(pair[0], mask=rec['mask'])[:, :, start:end].mean(axis=-1)
                r2 = rec['resp'].extract_epoch(pair[1], mask=rec['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r2 = (r2 - m) / sd

                tdr = TDR()
                tdr.fit(r1, r2)
                pair_tdr_weights = tdr.weights

                # ================================= active data ======================================
                r1 = rec['resp'].extract_epoch(pair[0], mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r2 = rec['resp'].extract_epoch(pair[1], mask=ra['mask'])[:, :, start:end].mean(axis=-1)
                r1 = (r1 - m) / sd
                r2 = (r2 - m) / sd

                # using overall tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, True, False, True, 
                            idx, snr1, snr2, cat_cat, tar_tar, cat_tar, ref_tar, ref_ref, ref_cat, aref_tar, aref_cat, aref_ref, atar_ref, atar_aref, atar_cat,
                            f1, f2, di, all_tdr_weights], \
                            index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair',
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 
                                'ref_tar', 'ref_ref', 'ref_cat', 'aref_tar', 'aref_cat', 'aref_ref', 'atar_ref','atar_aref', 'atar_cat',
                                'f1', 'f2', 'DI', 'dr_weights']).T)

                
                # using pair-specific tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, False, False, True, 
                            idx, snr1, snr2, cat_cat, tar_tar, cat_tar, ref_tar, ref_ref, ref_cat, aref_tar, aref_cat, aref_ref, atar_ref, atar_aref, atar_cat,
                            f1, f2, di, pair_tdr_weights], \
                            index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair',
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 
                                'ref_tar', 'ref_ref', 'ref_cat', 'aref_tar', 'aref_cat', 'aref_ref', 'atar_ref','atar_aref', 'atar_cat',
                                'f1', 'f2', 'DI', 'dr_weights']).T)

                # using PCA
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, False, True, True, 
                            idx, snr1, snr2, cat_cat, tar_tar, cat_tar, ref_tar, ref_ref, ref_cat, aref_tar, aref_cat, aref_ref, atar_ref, atar_aref, atar_cat,
                            f1, f2, di, pc_axes], \
                            index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair',
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar',
                                'ref_tar', 'ref_ref', 'ref_cat', 'aref_tar', 'aref_cat', 'aref_ref', 'atar_ref','atar_aref', 'atar_cat',
                                'f1', 'f2', 'DI', 'dr_weights']).T)
            
            
                # ================================= passive data ======================================
                r1 = rec['resp'].extract_epoch(pair[0], mask=rp['mask'])[:, :, start:end].mean(axis=-1)
                r2 = rec['resp'].extract_epoch(pair[1], mask=rp['mask'])[:, :, start:end].mean(axis=-1)

                # using overall tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(all_tdr_weights.T).T, r2.dot(all_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, True, False, False, 
                            idx, snr1, snr2, cat_cat, tar_tar, cat_tar, ref_tar, ref_ref, ref_cat, aref_tar, aref_cat, aref_ref, atar_ref, atar_aref, atar_cat,
                            f1, f2, di, all_tdr_weights], \
                            index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair',
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 
                                'ref_tar', 'ref_ref', 'ref_cat', 'aref_tar', 'aref_cat', 'aref_ref', 'atar_ref','atar_aref', 'atar_cat',
                                'f1', 'f2', 'DI', 'dr_weights']).T)
                
                # using pair-specific tdr
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pair_tdr_weights.T).T, r2.dot(pair_tdr_weights.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, False, False, False, 
                            idx, snr1, snr2, cat_cat, tar_tar, cat_tar, ref_tar, ref_ref, ref_cat, aref_tar, aref_cat, aref_ref, atar_ref, atar_aref, atar_cat,
                            f1, f2, di, pair_tdr_weights], \
                            index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair',
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 
                                'ref_tar', 'ref_ref', 'ref_cat', 'aref_tar', 'aref_cat', 'aref_ref', 'atar_ref','atar_aref', 'atar_cat',
                                'f1', 'f2', 'DI', 'dr_weights']).T)

                # using PCA
                dp, wopt, evals, evecs, evec_sim, dU = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T)
                dp_diag, _, _, _, _, _ = compute_dprime(r1.dot(pc_axes.T).T, r2.dot(pc_axes.T).T, diag=True)
                df = df.append(pd.DataFrame(data=[dp, wopt, evecs, evals, evec_sim, dU, dp_diag, False, True, False, 
                            idx, snr1, snr2, cat_cat, tar_tar, cat_tar, ref_tar, ref_ref, ref_cat, aref_tar, aref_cat, aref_ref, atar_ref, atar_aref, atar_cat,
                            f1, f2, di, pc_axes], \
                            index=['dp_opt', 'wopt', 'evecs', 'evals', 'evec_sim', 'dU', 'dp_diag', 'tdr_overall', 'pca', 'active', 'pair',
                                'snr1', 'snr2', 'cat_cat', 'tar_tar', 'cat_tar', 
                                'ref_tar', 'ref_ref', 'ref_cat', 'aref_tar', 'aref_cat', 'aref_ref', 'atar_ref','atar_aref', 'atar_cat',
                                'f1', 'f2', 'DI', 'dr_weights']).T)

            df['site'] = site
            if batch in [302, 307, 324]: area='A1'
            else: area='PEG'
            df['area'] = area
            df['batch'] = batch

            dfs.append(df)

df = pd.concat(dfs)
dtypes = {
    'dp_opt': 'float32',
    'wopt': 'object',
    'evecs': 'object',
    'evals': 'object',
    'evec_sim': 'float32',
    'dU': 'object',
    'dp_diag': 'float32',
    'tdr_overall': 'bool',
    'pca': 'bool',
    'active': 'bool',
    'pair': 'object',
    'snr1': 'float32',
    'snr2': 'float32',
    'cat_cat': 'bool',
    'tar_tar': 'bool',
    'cat_tar': 'bool',
    'ref_ref': 'bool',
    'ref_tar': 'bool',
    'ref_cat': 'bool',
    'aref_tar': 'bool',
    'aref_cat': 'bool',
    'aref_ref': 'bool',
    'f1': 'int32',
    'f2': 'int32',
    'DI': 'float32',
    'dr_weights': 'object',
    'batch': 'float32'
    }
dtypes_new = {k: v for k, v in dtypes.items() if k in df.columns}
df = df.astype(dtypes_new)

if zscore:
    if regress_pupil & regress_task:
        df.to_pickle(res_path + 'res_zscore_pr_br.pickle')
    elif regress_task:
        df.to_pickle(res_path + 'res_zscore_br.pickle')
    elif regress_pupil:
        df.to_pickle(res_path + 'res_zscore_pr.pickle')
    else:
        df.to_pickle(res_path + 'res_zscore.pickle')
else:
    if regress_pupil & regress_task:
        df.to_pickle(res_path + 'res_pr_br.pickle')
    elif regress_task:
        df.to_pickle(res_path + 'res_br.pickle')
    elif regress_pupil:
        df.to_pickle(res_path + 'res_pr.pickle')
    else:
        df.to_pickle(res_path + 'res.pickle')
plt.close('all')