"""
Copied from SVD. 04/28/2020
"""
import os
import sys
import pandas as pd
import scipy.signal as ss
import scipy.stats as st

import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems.db as nd
import nems_db.params

import nems.recording as recording
import nems.epoch as ep
import nems.plots.api as nplt
import nems.modelspec as ms

def get_model_results_per_state_model(batch=307, state_list=None,
                      loader = "psth.fs20.pup-ld-",
                      fitter = "_jk.nf20-basic",
                      basemodel = "-ref-psthfr.s_sdexp.S"):
    """
    loader = "psth.fs20.pup-ld-"
    fitter = "_jk.nf20-basic"
    basemodel = "-ref-psthfr.s_sdexp.S"
    state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

    d=get_model_results_per_state_model(batch=307, state_list=state_list,
                                        loader=loader,fitter=fitter,
                                        basemodel=basemodel)

    state_list defaults to
       ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    """

    if state_list is None:
        state_list = ['st.pup0.beh0', 'st.pup0.beh',
                      'st.pup.beh0', 'st.pup.beh']

    modelnames = [loader + s + basemodel + fitter for s in state_list]

    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    isolation = [nd.get_isolation(cellid=c, batch=batch).loc[0, 'min_isolation'] for c in cellids]

    if state_list[-1].endswith('fil') or state_list[-1].endswith('pas'):
        include_AP = True
    else:
        include_AP = False

    d = pd.DataFrame(columns=['cellid', 'modelname', 'state_sig',
                              'state_chan', 'MI', 'isolation',
                              'r', 'r_se', 'd', 'g', 'sp', 'state_chan_alt'])

    new_sdexp = False
    for mod_i, m in enumerate(modelnames):
        print('Loading modelname: ', m)
        modelspecs = nems_db.params._get_modelspecs(cellids, batch, m, multi='mean')

        for modelspec in modelspecs:
            meta = ms.get_modelspec_metadata(modelspec)
            phi = list(modelspec[0]['phi'].keys())
            c = meta['cellid']
            iso = isolation[cellids.index(c)]
            state_mod = meta['state_mod']
            state_mod_se = meta['se_state_mod']
            state_chans = meta['state_chans']
            if 'g' in phi:
                dc = modelspec[0]['phi']['d']
                gain = modelspec[0]['phi']['g']
            elif ('amplitude_g' in phi) & ('amplitude_d' in phi):
                new_sdexp = True
                dc = None
                gain = None
                g_amplitude = modelspec[0]['phi']['amplitude_g']
                g_base = modelspec[0]['phi']['base_g']
                g_kappa = modelspec[0]['phi']['kappa_g']
                g_offset = modelspec[0]['phi']['offset_g']
                d_amplitude = modelspec[0]['phi']['amplitude_d']
                d_base = modelspec[0]['phi']['base_d']
                d_kappa = modelspec[0]['phi']['kappa_d']
                d_offset = modelspec[0]['phi']['offset_d']

            gain_mod = None
            dc_mod = None
            if 'state_mod_gain' in meta.keys():
                gain_mod = meta['state_mod_gain']
                dc_mod = meta['state_mod_dc']

            if dc is not None:
                sp = modelspec[0]['phi'].get('sp', np.zeros(gain.shape))
                if dc.ndim > 1:
                    dc = dc[0, :]
                    gain = gain[0, :]
                    sp = sp[0, :]

            a_count = 0
            p_count = 0

            for j, sc in enumerate(state_chans):
                if gain is not None:
                    gain_val = gain[j]
                    dc_val = dc[j]
                    sp_val = sp[j]
                else:
                    gain_val = None
                    dc_val = None
                    sp_val = None
                r = {'cellid': c, 'state_chan': sc, 'modelname': m,
                     'isolation': iso,
                     'state_sig': state_list[mod_i],
                     'g': gain_val, 'd': dc_val, 'sp': sp_val,
                     'MI': state_mod[j],
                     'r': meta['r_test'][0], 'r_se': meta['se_test'][0]}
                if new_sdexp:
                    r.update({'g_amplitude': g_amplitude[0, j], 'g_base': g_base[0, j], 'g_kappa': g_kappa[0, j], 'g_offset': g_offset[0, j],
                                'd_amplitude': d_amplitude[0, j], 'd_base': d_base[0, j], 'd_kappa': d_kappa[0, j], 'd_offset': d_offset[0, j]})
                if gain_mod is not None:
                    r.update({'gain_mod': gain_mod[j], 'dc_mod': dc_mod[j]})

                d = d.append(r, ignore_index=True)
                l = len(d) - 1

                if include_AP and sc.startswith("FILE_"):
                    siteid = c.split("-")[0]
                    fn = "%" + sc.replace("FILE_","") + "%"
                    sql = "SELECT * FROM gDataRaw WHERE cellid=%s" +\
                       " AND parmfile like %s"
                    dcellfile = nd.pd_query(sql, (siteid, fn))
                    if dcellfile.loc[0]['behavior'] == 'active':
                        a_count += 1
                        d.loc[l,'state_chan_alt'] = "ACTIVE_{}".format(a_count)
                    else:
                        p_count += 1
                        d.loc[l,'state_chan_alt'] = "PASSIVE_{}".format(p_count)
                else:
                    d.loc[l,'state_chan_alt'] = d.loc[l,'state_chan']

    #d['r_unique'] = d['r'] - d['r0']
    #d['MI_unique'] = d['MI'] - d['MI0']

    return d