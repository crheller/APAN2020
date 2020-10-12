"""
For each target, time bin, cache noise correlations and behavioral DI
"""

from nems_lbhb.baphy_experiment import BAPHYExperiment
import charlieTools.noise_correlations as nc
import charlieTools.preprocessing as preproc
import nems_lbhb.tin_helpers as thelp
import nems.db as nd
import pandas as pd
import numpy as np

res_path = '/home/charlie/Desktop/lbhb/code/projects/APAN2020/results/'

time_bins = [
    (0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.1, 0.3),
    (0.2, 0.4)
]

options = {'resp': True, 'pupil': True, 'rasterfs': 10}
batches = [324, 325]
recache = False
regress_pupil = False  # regress out first order pupil

dfs = []
for batch in batches:
    sites = np.unique([c[:7] for c in nd.get_batch_cells(batch).cellid])
    sites = [s for s in sites if s!='CRD013b']
    for site in sites:
        manager = BAPHYExperiment(batch=batch, siteid=site)
        rec = manager.get_recording(recache=recache, **options)
        rec['resp'] = rec['resp'].rasterize()

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

        # for each epoch / time bin, compute active / passive noise correlations
        # save rsc, snr, f, active/passive state

        for epoch in targets + catch:
            if 'TAR_' in epoch:
                di = behavior_performance['LI'][epoch.strip('TAR_').strip('CAT_')+'_'+catch[0].strip('CAT_')]
                diall = behavior_performance_all['LI'][epoch.strip('TAR_').strip('CAT_')+'_'+catch[0].strip('CAT_')]
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

                df['snr'] = thelp.get_snrs([epoch])[0]
                df['f'] = thelp.get_tar_freqs([epoch])[0]
                df['tbin'] = '_'.join([str(t) for t in tb])
                df['DI'] = di
                df['DIall'] = diall
                df['DIref'] = diref
                df['DIrefall'] = direfall
                df['site'] = site
                if batch==324: area='A1'
                else: area='PEG'
                df['area'] = area

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
    'site': 'object'
    }
dtypes_new = {k: v for k, v in dtypes.items() if k in dfall.columns}
dfall = dfall.astype(dtypes_new)

# save results
if regress_pupil:
    dfall.to_pickle(res_path + 'rsc_df_pr.pickle')
else:
    dfall.to_pickle(res_path + 'rsc_df.pickle')