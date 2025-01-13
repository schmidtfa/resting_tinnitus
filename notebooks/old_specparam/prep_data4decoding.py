#%%
import pandas as pd
from pathlib import Path
import joblib
import numpy as np
import scipy.stats as stats

# %%

all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam_3').glob(f'*/*__peak_threshold_2__freq_range_[[]0.25, 98[]].dat'))
# %%
periodic, aperiodic = [], []

for f in all_files:
    
    cur_data = joblib.load(f)

    periodic.append(cur_data['periodic'])
    aperiodic.append(cur_data['aperiodic'])

#%% 
physio = ['ECG', 'EOGV', 'EOGH', '???']
df_periodic = pd.concat(periodic).query('ch_name != @physio').reset_index()
df_aperiodic = pd.concat(aperiodic).query('ch_name != @physio').reset_index()

#% mask bad fits (not using the pandas mask on purpose to avoid biases later)
mask_level = 0.9
df_aperiodic[['offset', 'knee', 'exponent']] = df_aperiodic[['offset', 'knee', 'exponent']].mask(df_aperiodic['r_squared'] < mask_level)
df_periodic[['delta','theta','alpha', 'beta', 'gamma']] =  df_periodic[['delta','theta','alpha', 'beta', 'gamma']].mask(df_periodic['r_squared'] < mask_level)

#%% no real improvement by knee fit. plus knees are mostly in vlf 
df_aperiodic = df_aperiodic.query('aperiodic_mode == "fixed"')
df_periodic = df_periodic.query('aperiodic_mode == "fixed"')


#%%
peak_params = ['cf', 'bw', 'pw'] #this order matters to get the bad beta indices

all_p_dfs = []

df_periodic

#TODO: Add good comments
for cu_p_param in peak_params:

    p_cols_of_interest = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'line_noise', 'subject_id', 'ch_name', 'tinnitus', 'n_peaks']

    df_p_cut = df_periodic.query(f'peak_params == "{cu_p_param}"')[p_cols_of_interest]

    if cu_p_param == 'cf':
        #remove train and line noise from n peaks
        #remove line noise peaks from total peak count
        df_p_cut['n_peaks'] = df_p_cut['n_peaks'] - (np.isnan(df_p_cut['line_noise']) == False).to_numpy().astype(int) 
        # remove beta peaks that may could be caused by the train
        df_p_cut['n_peaks'] = df_p_cut['n_peaks'] - np.logical_and(df_p_cut['beta'] < 17, df_p_cut['beta'] > 16).to_numpy().astype(int) 
        bad_beta_indices = np.logical_and(df_p_cut['beta'] < 17, df_p_cut['beta'] > 16).to_numpy()
        df_p_cut['beta'][bad_beta_indices] = np.nan

        #get oscillations
        df_osc_present = (np.isnan(df_p_cut[['delta', 'theta', 'alpha', 'beta', 'gamma']]) == False).astype(int)
        df_osc_present[['subject_id', 'ch_name', 'tinnitus']] = df_p_cut[['subject_id', 'ch_name', 'tinnitus']] #this should be ok
        #transform 
        dfs_tmp = []
        for col in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
           dfs_tmp.append(df_p_cut[col].fillna(df_p_cut.groupby('ch_name')[col].transform('mean')))
        
        df_p_cut_clean = pd.concat(dfs_tmp, axis=1)
        df_p_cut_clean[['subject_id', 'ch_name', 'tinnitus', 'n_peaks']] = df_p_cut[['subject_id', 'ch_name', 'tinnitus', 'n_peaks']] #this should be ok
           
    else:
        df_p_cut['beta'][bad_beta_indices] = np.nan
        df_p_cut_clean = df_p_cut.fillna(0) #if no relevant band peaks are there replace with 0
        df_p_cut_clean.drop(columns='n_peaks', inplace=True)
        df_p_cut_clean.drop(columns='line_noise', inplace=True)

    df_p_cut_clean.rename(columns={'delta': 'delta' + f'_{cu_p_param}',
                                   'theta': 'theta' + f'_{cu_p_param}',
                                   'alpha': 'alpha' + f'_{cu_p_param}',
                                   'beta': 'beta' + f'_{cu_p_param}',
                                   'gamma': 'gamma' + f'_{cu_p_param}'
                        }, inplace=True)
    
    
    all_p_dfs.append(df_p_cut_clean)


df_p = all_p_dfs[0].merge(all_p_dfs[1], on=['subject_id', 'ch_name', 'tinnitus']).merge(all_p_dfs[2], on=['subject_id', 'ch_name', 'tinnitus'])
df_osc_present.rename(columns={'delta': 'delta_osc',
                               'theta': 'theta_osc',
                               'alpha': 'alpha_osc',
                               'beta': 'beta_osc',
                               'gamma': 'gamma_osc'
                        }, inplace=True)


#%%
cols_of_interest = ['offset', 'exponent', 'knee_freq', 'subject_id', 'ch_name', 'tinnitus']

df_aperiodic_cut = df_aperiodic[cols_of_interest]

df_ap = df_aperiodic_cut.drop(columns='knee_freq')


dfs_tmp = []
for col in ['offset', 'exponent']:
    dfs_tmp.append(df_ap[col].fillna(df_ap.groupby('ch_name')[col].transform('mean')))

df_ap_cut_clean = pd.concat(dfs_tmp, axis=1)

df_ap_cut_clean[['subject_id', 'ch_name', 'tinnitus']] = df_ap[['subject_id', 'ch_name', 'tinnitus']] #this should be ok
           

#%%
df_cmb = df_ap_cut_clean.merge(df_p, on=['subject_id', 'ch_name', 'tinnitus']).merge(df_osc_present, on=['subject_id', 'ch_name', 'tinnitus'])

#%%
df_cmb.to_csv('../data/tinnitus_all_spec_features.csv')

# %%



# for cu_p_param in peak_params:

#     # 'gamma','delta', 'theta'
#     p_cols_of_interest = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'line_noise', 'subject_id', 'ch_name', 'tinnitus', 'n_peaks']

#     df_p_cut = drop_knee(df_periodic.query(f'peak_params == "{cu_p_param}"'))[p_cols_of_interest]

#     if cu_p_param == 'cf':
#         #remove train and line noise from n peaks
#         df_p_cut['n_peaks'] = df_p_cut['n_peaks'] - (np.isnan(df_p_cut['line_noise']) == False).to_numpy().astype(int)
#         df_p_cut['n_peaks'] = df_p_cut['n_peaks'] - np.logical_and(df_p_cut['beta'] < 17, df_p_cut['beta'] > 16).to_numpy().astype(int)
#         bad_beta_indices = np.logical_and(df_p_cut['beta'] < 17, df_p_cut['beta'] > 16).to_numpy()
#         df_p_cut['beta'][bad_beta_indices] = np.nan

#         df_osc_present = (np.isnan(df_p_cut[['delta', 'theta', 'alpha', 'beta', 'gamma']]) == False).astype(int)
#         df_osc_present[['subject_id', 'ch_name', 'tinnitus']] = df_p_cut[['subject_id', 'ch_name', 'tinnitus']]

#         df_p_cut_clean = df_p_cut.fillna(df_p_cut.mean())    
#     else:
#         df_p_cut['beta'][bad_beta_indices] = np.nan
#         df_p_cut_clean = df_p_cut.fillna(0) #if no relevant band peaks are there replace with 0
#         df_p_cut_clean.drop(columns='n_peaks', inplace=True)

#     df_p_cut_clean.rename(columns={'delta': 'delta' + f'_{cu_p_param}',
#                                    'theta': 'theta' + f'_{cu_p_param}',
#                                    'alpha': 'alpha' + f'_{cu_p_param}',
#                                    'beta': 'beta' + f'_{cu_p_param}',
#                                    'gamma': 'gamma' + f'_{cu_p_param}'
#                         }, inplace=True)
    
    
#     all_p_dfs.append(df_p_cut_clean)




# df_p = all_p_dfs[0].merge(all_p_dfs[1], on=['subject_id', 'ch_name', 'tinnitus']).merge(all_p_dfs[2], on=['subject_id', 'ch_name', 'tinnitus'])
# df_osc_present.rename(columns={'delta': 'delta_osc',
#                                'theta': 'theta_osc',
#                                'alpha': 'alpha_osc',
#                                'beta': 'beta_osc',
#                                'gamma': 'gamma_osc'
#                         }, inplace=True)



# def drop_knee(cur_df, no_knee=True):
#     #cur_df = knee_or_fixed(cur_df) #get fixed or knee model

#     if no_knee:
#        cur_df = cur_df.query('aperiodic_mode == "fixed"')
#     else:
        
#         knee_settings = joblib.load('/mnt/obob/staff/fschmidt/resting_tinnitus/data/knee_settings.dat')
#         knee_chans = knee_settings['knee']
#         fixed_chans = knee_settings['fixed']
#         cur_df = pd.concat([cur_df.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
#                             cur_df.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])


#     return cur_df
# # %%
# test_columns_aperiodic = ['offset', 'knee', 'exponent']
# already_nan = {col: np.isnan(df_aperiodic[col]).sum() for col in test_columns_aperiodic}
# new_nan = {col: np.logical_and(aperiodic_mask, np.isnan(df_aperiodic[col]) == False).sum() for col in test_columns_aperiodic}
# should_be_nan = {col: already_nan[col] + new_nan[col] for col in test_columns_aperiodic}

# df_aperiodic[df_aperiodic['r_squared'] < mask_level][['offset', 'knee', 'exponent']] = np.nan
# df_periodic[df_periodic['r_squared'] < mask_level][['delta','theta','alpha', 'beta', 'gamma']] = np.nan

# np.isnan(df_aperiodic['exponent']).sum()

