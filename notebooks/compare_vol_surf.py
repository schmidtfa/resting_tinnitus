#%%
import numpy as np
import joblib
import pandas as pd
from pyrasa.utils.aperiodic_utils import compute_aperiodic_model
from pyrasa.utils.peak_utils import get_peak_params, get_band_info
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
sns.set_context('talk')
from pathlib import Path
import scipy.stats as stats
# %%
base_path = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'
all_files_vol = list(Path(base_path).glob(f'*/*_irasa__source_volume.dat'))
all_files_surf = list(Path(base_path).glob(f'*/*_irasa__source_surface.dat'))
# %%
all_files = list(all_files_surf)
fit_func = 'knee'
all_df_l, surf_avgs, vol_avgs = [], [], []


for file_surf, file_vol in zip(all_files_surf, all_files_vol):

    cur_d_surf = joblib.load(file_surf)
    cur_d_vol = joblib.load(file_vol)

    lh_idx = cur_d_surf['src']['label_info']['lh']
    rh_idx = cur_d_surf['src']['label_info']['rh']
    labels_surf = cur_d_surf['src']['label_info']['names_order_mne']
    labels_surf_clean = np.array([('lh_' + labels_surf[idx]) if logi == True else ('rh_' + labels_surf[idx]) for idx, logi in enumerate(lh_idx)])
    data_surf_clean = cur_d_surf['src']['label_tc_ap']
    
    ctx_labels = np.array([label for label in cur_d_vol['src']['label_info']['ctx_labels']])
    ctx_labels = [label[:3] + 'Unknown' if 'Medial' in label else label for label in ctx_labels]
    ctx_logical = cur_d_vol['src']['label_info']['ctx_logical']
    ctx_data = cur_d_vol['src']['label_tc_ap'][ctx_logical,:]
    
    trans_array = np.array([np.where(labels_surf_clean == label)[0][0] for label in ctx_labels])

    params_surf = compute_aperiodic_model(data_surf_clean[trans_array,:], 
                                        cur_d_surf['freq'], 
                                        fit_func=fit_func, 
                                        ch_names=ctx_labels,
                                        scale=True).aperiodic_params
    params_surf['src_type'] = 'surface'
    surf_avgs.append(np.mean(data_surf_clean[trans_array,:], axis=0))

    params_vol = compute_aperiodic_model(ctx_data, 
                                        cur_d_surf['freq'], 
                                        fit_func=fit_func, 
                                        ch_names=ctx_labels,
                                        scale=True).aperiodic_params
    params_vol['src_type'] = 'volume'
    vol_avgs.append(np.mean(ctx_data, axis=0))
    
    params_cmb = pd.concat([params_vol, params_surf])
    params_cmb['subject_id'] = cur_d_surf['subject_id']

    all_df_l.append(params_cmb)
# %%
all_df = pd.concat(all_df_l)

#%%
df_surf = pd.DataFrame(surf_avgs).T
df_surf['freqs'] = cur_d_surf['freq']

df_vol = pd.DataFrame(vol_avgs).T
df_vol['freqs'] = cur_d_vol['freq']

f, ax = plt.subplots(figsize=(4,4))
sns.lineplot(data=df_vol.melt(id_vars='freqs').query('freqs < 60'), x='freqs', y='value', ax=ax)
sns.lineplot(data=df_surf.melt(id_vars='freqs').query('freqs < 60'), x='freqs', y='value', ax=ax)
ax.set_xscale('log')
ax.set_yscale('log')
# %%

f, axes = plt.subplots(ncols=4 , figsize=(16,4))

for ix, feature in enumerate(['Offset', 'Exponent_1', 'Knee Frequency (Hz)', 'Exponent_2']):

    all_df_pv = all_df.pivot_table(values=feature, index=['subject_id', 'ch_name'], columns='src_type').reset_index()

    corr_by_area = all_df_pv.groupby('ch_name')[['surface', 'volume']].corr().reset_index().query('src_type == "surface"')
    
    axes[ix].hist((corr_by_area['volume']))
    axes[ix].set_xlabel('Correlation by Area \n (Surface x Volume)')
    axes[ix].set_xlim(0, 1)
    axes[ix].set_title(feature)
    sns.despine()
# %%
