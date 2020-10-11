"""
Load / plot psth model results. 
Split by waveform type. Are behavior effects bigger in one cell type?
"""

from SingleCell.mod_per_state import get_model_results_per_state_model
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

recache = True

path = '/auto/users/hellerc/results/APAN2020/first_order_model_results/'

state_list = ['st.pup0.beh0','st.pup.beh0', 'st.pup0.beh', 'st.pup.beh']
basemodel2 = "-ref-psthfr_sdexp.S"
loader = "psth.fs20.pup-ld-"
fitter = '_jk.nf20-basic'
batches = [324, 325]
if recache:
    for batch in batches:
        d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                            basemodel=basemodel2, loader=loader, fitter=fitter)
        d.to_csv(os.path.join(path, 'd_{}_pup_beh_sdexp.csv'.format(batch)))
    
A1 = pd.read_csv(os.path.join(path,'d_324_pup_beh_sdexp.csv'), index_col=0)
PEG = pd.read_csv(os.path.join(path,'d_325_pup_beh_sdexp.csv'), index_col=0)
A1['area'] = 'A1'
PEG['area'] = 'PEG'
df = pd.concat([A1, PEG])

try:
    df['r'] = [np.float(r.strip('[]')) for r in df['r'].values]
    df['r_se'] = [np.float(r.strip('[]')) for r in df['r_se'].values]
except:
    pass

df = df[df.state_chan=='active'].pivot(columns='state_sig', index='cellid', values=['gain_mod', 'dc_mod', 'MI', 'r', 'r_se', 'area'])
dc = df.loc[:, pd.IndexSlice['dc_mod', 'st.pup.beh']] - df.loc[:, pd.IndexSlice['dc_mod', 'st.pup.beh0']]
gain = df.loc[:, pd.IndexSlice['gain_mod', 'st.pup.beh']] - df.loc[:, pd.IndexSlice['gain_mod', 'st.pup.beh0']]
sig = (df.loc[:, pd.IndexSlice['r', 'st.pup.beh']] - df.loc[:, pd.IndexSlice['r', 'st.pup.beh0']]) > \
            (df.loc[:, pd.IndexSlice['r_se', 'st.pup.beh']] + df.loc[:, pd.IndexSlice['r_se', 'st.pup.beh0']])
nsig = sig.sum()
ntot = gain.shape[0]

