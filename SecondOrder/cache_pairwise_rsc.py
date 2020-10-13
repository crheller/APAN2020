"""
For each target, time bin, cache noise correlations and behavioral DI
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.noise_correlations as nc
import charlieTools.preprocessing as preproc
import nems_lbhb.tin_helpers as thelp
from nems_lbhb.baphy import parse_cellid
import nems.db as nd
import pandas as pd
import numpy as np
from charlieTools.ptd_ms.utils import which_rawids

res_path = '/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/'

batches = [302, 307, 324, 325]
Aoptions = dict.fromkeys(batches)
Aoptions[302] = {'resp': True, 'pupil': True, 'rasterfs': 10}
Aoptions[307] = {'resp': True, 'pupil': True, 'rasterfs': 20}
Aoptions[324] = {'resp': True, 'pupil': True, 'rasterfs': 10}
Aoptions[325] = {'resp': True, 'pupil': True, 'rasterfs': 10}

twin = {
    302:[
        (0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.1, 0.3),
        (0.2, 0.4)
    ],
    307:[
        (0.05, 0.15),
        (0.15, 0.25),
        (0.25, 0.35),
        (0.35, 0.45),
        (0.45, 0.55),
        (0.55, 0.65),
        (0.65, 0.75),
        (0.75, 0.85),
        (0.85, 0.95),
        (1.05, 1.15)
    ]
}
twin[324] = twin[302]
twin[325] = twin[302]

recache = False
regress_pupil = True  # regress out first order pupil

dfs = []
for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if s!='CRD013b']
    options = Aoptions[batch]
    time_bins = twin[batch]
    sites = [s for s in sites if (s!='CRD013b') & ('gus' not in s)]
    if batch == 302:
        sites1 = [s+'.e1:64' for s in sites]
        sites2 = [s+'.e65:128' for s in sites]
        sites = sites1 + sites2
    for site in sites:
        if batch == 307:
            rawid = which_rawids(site)
        else:
            rawid = None
        manager = BAPHYExperiment(batch=batch, siteid=site[:7], rawid=rawid)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()
        if batch == 302:
            c, _ = parse_cellid({'cellid': site, 'batch': batch})
            rec['resp'] = rec['resp'].extract_channels(c)

        behavior_performance = manager.get_behavior_performance(**options)
        options['keep_following_incorrect_trial'] = True
        options['keep_cue_trials'] = True
        options['keep_early_trials'] = True
        behavior_performance_all = manager.get_behavior_performance(**options)

        # regress out first order pupil
        if regress_pupil:
            rec = preproc.regress_state(rec, state_sigs=['pupil'])

        ra = rec.copy()
        ra = ra.create_mask(True)
        if batch in [324, 325]:
            ra = ra.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL'])
        elif batch == 302:
            ra = ra.and_mask(['HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL'])
        elif batch == 307:
            ra = ra.and_mask(['HIT_TRIAL'])

        rp = rec.copy()
        rp = rp.create_mask(True)
        rp = rp.and_mask(['PASSIVE_EXPERIMENT'])

        # find / sort epoch names
        if batch in [324, 325]:
            targets = thelp.sort_targets([f for f in ra['resp'].epochs.name.unique() if 'TAR_' in f])
            targets = [t for t in targets if (ra['resp'].epochs.name==t).sum()>=5]
            on_center = thelp.get_tar_freqs([f.strip('REM_') for f in ra['resp'].epochs.name.unique() if 'REM_' in f])[0]
            targets = [t for t in targets if str(on_center) in t]
            catch = [f for f in ra['resp'].epochs.name.unique() if 'CAT_' in f]
            catch = [c for c in catch if str(on_center) in c]
            targets_str = targets
            catch_str = catch
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
            targets_str = [f'TAR_{t}+{snr}dB+Noise' for snr, t in zip(snrs, tf)]
            targets_str = targets_str[::-1]
            targets = targets[::-1]
            # only keep targets w/ at least 5 reps in active
            targets_str = [ts for t, ts in zip(targets, targets_str) if (ra['resp'].epochs.name==t).sum()>=5]
            targets = [t for t in targets if (ra['resp'].epochs.name==t).sum()>=5]
            catch = []
            catch_str = []
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

        # for each epoch / time bin, compute active / passive noise correlations
        # save rsc, snr, f, active/passive state

        for epoch, epoch_str in zip(targets + catch, targets_str + catch_str):
            if 'TAR_' in epoch_str:
                if batch in [324, 325]:
                    di = behavior_performance['LI'][epoch.strip('TAR_').strip('CAT_')+'_'+catch[0].strip('CAT_')]
                    diall = behavior_performance_all['LI'][epoch.strip('TAR_').strip('CAT_')+'_'+catch[0].strip('CAT_')]
                elif batch == 302:
                    di = behavior_performance['LI'][epoch.strip('TAR_').strip('CAT_')+'_'+catch[0].strip('TAR_')]
                    diall = behavior_performance_all['LI'][epoch.strip('TAR_').strip('CAT_')+'_'+catch[0].strip('TAR_')]
                elif batch == 307:
                    # not really an explicit "catch" for this data
                    di = np.inf
                    diall = np.inf
                diref = behavior_performance['DI'][epoch.strip('TAR_').strip('CAT_')]
                direfall = behavior_performance_all['DI'][epoch.strip('TAR_').strip('CAT_')]
            else:
                di = np.inf
                diall = np.inf
            for tb in time_bins:
                sidx = int(tb[0] * options['rasterfs']) 
                eidx = int(tb[1] * options['rasterfs']) 
                da = {k: r[:, :, sidx:eidx] for k, r in rec['resp'].extract_epochs([epoch], mask=ra['mask']).items()}
                dp = {k: r[:, :, sidx:eidx] for k, r in rec['resp'].extract_epochs([epoch], mask=rp['mask']).items()}

                dfa = nc.compute_rsc(da, chans=rec['resp'].chans).rename(columns={'rsc': 'active', 'pval': 'pa'})
                dfp = nc.compute_rsc(dp, chans=rec['resp'].chans).rename(columns={'rsc': 'passive', 'pval': 'pp'})
                df = pd.concat([dfa, dfp], axis=1)

                df['snr'] = thelp.get_snrs([epoch_str])[0]
                df['f'] = thelp.get_tar_freqs([epoch])[0]
                df['tbin'] = '_'.join([str(t) for t in tb])
                df['DI'] = di
                df['DIall'] = diall
                df['DIref'] = diref
                df['DIrefall'] = direfall
                df['site'] = site
                if batch in [324, 302, 307]: area='A1'
                else: area='PEG'
                df['area'] = area
                df['batch'] = batch

                dfs.append(df)


dfall = pd.concat(dfs)
dtypes = {
    'active': 'float32',
    'passive': 'float32',
    'pa': 'float32',
    'pp': 'float32',
    'snr': 'float32',
    'f': 'float32',
    'tbin': 'object',
    'DI': 'float32',
    'DIall': 'float32',
    'DIref': 'float32',
    'DIrefall': 'float32',
    'area': 'object',
    'site': 'object',
    'batch': 'float32'
    }
dtypes_new = {k: v for k, v in dtypes.items() if k in dfall.columns}
dfall = dfall.astype(dtypes_new)

# save results
if regress_pupil:
    dfall.to_pickle(res_path + 'rsc_df_pr.pickle')
else:
    dfall.to_pickle(res_path + 'rsc_df.pickle')