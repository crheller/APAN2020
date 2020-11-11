"""
For an example site, look at relationship between choice axis(es), 
decoding axis, and noise correlation axis

Use leave-one-out subsampling of neurons to generate prob. distribution 
over angles between vectors, and compare to distribution of random vectors 
in that nD space to see if significantly more/less aligned than chance.
"""
from nems_lbhb.baphy_experiment import BAPHYExperiment

batch = 324
site = 'CRD016c'
rawid = None
recache = False
options = {'resp': True, 'pupil': True, 'rasterfs': 10}

tstart = int( 0.1 * options['rasterfs'] )
tend = int( 0.3 * options['rasterfs'] )

manager = BAPHYExperiment(batch=batch, siteid=site[:7], rawid=rawid)
rec = manager.get_recording(recache=recache, **options)
rec['resp'] = rec['resp'].rasterize()


# ===================== USE CATCH RESPONSES =======================
amask = ['CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL']
r = rec.copy()
r = r.and_mask(amask + ['PASSIVE_EXPERIMENT'])
ra = r.and_mask(amask)
rp = r.and_mask(['PASSIVE_EXPERIMENT'])
stim = [s for s in ra.apply_mask(reset_epochs=True).epochs.name.unique() if 'CAT_' in s]

da = r['resp'].extract_epoch(stim[0], mask=ra['mask'])[:, :, tstart:tend].mean(axis=-1)
dp = r['resp'].extract_epoch(stim[0], mask=rp['mask'])[:, :, tstart:tend].mean(axis=-1)