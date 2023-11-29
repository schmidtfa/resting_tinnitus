#%%
import pandas as pd
import numpy as np
import xarray as xr
import rsatoolbox as rsa
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

import seaborn as sns
sns.set_style('ticks')
sns.set_context('poster')
#%%
# what do we want to learn. 
# 1 )Are subjects with tinnitus more similar 
# to each other then subjects without tinnitus?
# 2) What features are most similar in subjects with tinnitus
# 3) Where in the brain are the most similarities?




#%%

df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
subject_list = list(df_all['subject_id'].unique())



all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam_full_spectra').glob(f'*/*fixed__peak_threshold_2__freq_range_[[]0.25, 98[]].dat'))
#all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam').glob(f'*/*__peak_threshold_2.5__freq_range_[[]0.25, 98[]].dat'))

all_cur_ids = [str(p).split('/')[-2] for p in all_files]

[p for p in subject_list if p not in all_cur_ids]

#%%
periodic, aperiodic = [], []

for f in all_files:
    
    cur_data = joblib.load(f)

    cur_df_ap = pd.DataFrame(cur_data['full_spectra'][2]['src']['aperiodic'])
    cur_df_ap['subject_id'] = str(f).split('/')[8]
    cur_df_ap['frequency'] = np.arange(0.25, 98.25, 0.25)
    cur_df_ap_t = cur_df_ap.melt(id_vars=['subject_id', 'frequency'], var_name='ch_name', value_name='power')

    cur_df_p = pd.DataFrame(cur_data['full_spectra'][2]['src']['periodic'])
    cur_df_p['subject_id'] = str(f).split('/')[8]
    cur_df_p['frequency'] = np.arange(0.25, 98.25, 0.25)
    cur_df_p_t = cur_df_p.melt(id_vars=['subject_id', 'frequency'], var_name='ch_name', value_name='power')

    periodic.append(cur_df_p_t)
    aperiodic.append(cur_df_ap_t)


#%%
feature_list = ['ch_name', 'subject_id', 'frequency', 'power']
ch_labels = cur_data['label_info']['names_order_mne'][2:]
freqs = np.arange(0.25, 98.25, 0.25)

df_p_t = np.squeeze(pd.concat(periodic)
                 .query('ch_name != "???"')
                 .merge(df_all[['subject_id', 'tinnitus']], on='subject_id')
                 .query('tinnitus == True')[feature_list]
                 .set_index(['ch_name', 'subject_id', 'frequency'])
                 .to_xarray()
                 .to_array()
                 .to_numpy())


df_p_no_t = np.squeeze(pd.concat(periodic)
                 .query('ch_name != "???"')
                 .merge(df_all[['subject_id', 'tinnitus']], on='subject_id')
                 .query('tinnitus == False')[feature_list]
                 .set_index(['ch_name', 'subject_id', 'frequency'])
                 .to_xarray()
                 .to_array()
                 .to_numpy())


df_ap_t = np.squeeze(pd.concat(aperiodic)
                 .query('ch_name != "???"')
                 .merge(df_all[['subject_id', 'tinnitus']], on='subject_id')
                 .query('tinnitus == True')[feature_list]
                 .set_index(['ch_name', 'subject_id', 'frequency'])
                 .to_xarray()
                 .to_array()
                 .to_numpy())


df_ap_no_t = np.squeeze(pd.concat(aperiodic)
                 .query('ch_name != "???"')
                 .merge(df_all[['subject_id', 'tinnitus']], on='subject_id')
                 .query('tinnitus == False')[feature_list]
                 .set_index(['ch_name', 'subject_id', 'frequency'])
                 .to_xarray()
                 .to_array()
                 .to_numpy())


#%%
def build_rsa_dataset(data_ct, data_tin, ch_labels=ch_labels, freqs=freqs):

       data = np.concatenate([data_ct, data_tin], axis=1).swapaxes(0, 1)

       tinnitus = np.concatenate([np.ones(data_ct.shape[1]) * 0, np.ones(data_tin.shape[1])])
       shape = data.shape
       data_sets = []
       for freq_ix in range(shape[2]):
                            
              obs_des = {
                     'subject': np.arange(shape[0]),
                     'tinnitus': tinnitus,
                     } # observation descriptor --> can be vigorously extended
              des = {
                     'freq': freqs[freq_ix]
                     }
              
              data_sets.append(rsa.data.Dataset(data[:,:,freq_ix],
                                                descriptors=des,
                                                obs_descriptors=obs_des,
                                                channel_descriptors={'names': ch_labels}
                                                )
                                                )

       return data_sets


def build_rsa_dataset_spatial(data_ct, data_tin, ch_labels=ch_labels, freqs=freqs):

       data = np.concatenate([data_ct, data_tin], axis=1).swapaxes(0, 1).swapaxes(1, 2)

       tinnitus = np.concatenate([np.ones(data_ct.shape[1]) * 0, np.ones(data_tin.shape[1])])
       shape = data.shape
       data_sets = []
       for ch_ix in range(shape[2]):
                            
              obs_des = {
                     'subject': np.arange(shape[0]),
                     'tinnitus': tinnitus,
                     } # observation descriptor --> can be vigorously extended
              des = {
                     'ch_index': ch_labels[ch_ix]
                     }
              
              data_sets.append(rsa.data.Dataset(data[:,:,ch_ix],
                                                descriptors=des,
                                                obs_descriptors=obs_des,
                                                channel_descriptors={'freq': freqs}
                                                )
                                                )

       return data_sets


data_sets_p = build_rsa_dataset(10**df_p_no_t, 10**df_p_t)
data_sets_ap = build_rsa_dataset(10**df_ap_no_t, 10**df_ap_t)

data_sets_p_s = build_rsa_dataset_spatial(10**df_p_no_t, 10**df_p_t)
data_sets_ap_s = build_rsa_dataset_spatial(10**df_ap_no_t, 10**df_ap_t)



#%%

def tril2nan(m):
   
   m_new = np.empty(m.shape)
   for freq in range(m.shape[0]):

       m_tmp = m[freq]
       m_tmp[np.triu_indices(m_tmp.shape[1], -1)] = np.nan

       m_new[freq, :, :] = m_tmp
   return m_new


#%%
def get_rdm(data_sets, freqs, ch_names, distance = 'euclidean', freq=True):

       rdms_data = rsa.rdm.calc_rdm(data_sets, 
                             method=distance, 
                             descriptor='subject'
                             )
       rdm = rdms_data.get_matrices()

       for cur_dim in rdm:
              np.fill_diagonal(cur_dim, np.nan)


       ave_dist_in = np.nanmean(tril2nan((rdm[:,53:,53:] + rdm[:,:51,:51]) /2), axis=2)
       ave_dist_cross = np.nanmean(tril2nan(rdm[:,:53, 51:]), axis=2)


       df_diss = pd.DataFrame(np.concatenate([ave_dist_in, ave_dist_cross], axis=1))

       if freq:
             df_diss['Frequency'] = freqs
             id_var = 'Frequency'
       else:
             df_diss['ch_name'] = ch_names
             id_var = 'ch_name'
       df_diss_melt = df_diss.melt(id_vars=id_var,
                            var_name='subject_ix',
                            value_name='Dissimilarity')
       df_diss_melt['tinnitus'] = np.where(df_diss_melt['subject_ix'] < 53, True, False)

       return df_diss_melt

rdm_p_f = get_rdm(data_sets_p, freqs=freqs, ch_names=ch_labels, freq=True)
rdm_ap_f = get_rdm(data_sets_ap, freqs=freqs, ch_names=ch_labels, freq=True)
rdm_p_s = get_rdm(data_sets_p_s, freqs=freqs, ch_names=ch_labels, freq=False)
rdm_ap_s = get_rdm(data_sets_ap_s, freqs=freqs, ch_names=ch_labels, freq=False)

#%%
f, axes = plt.subplots(figsize=(16, 8), ncols=2)

sns.lineplot(data=rdm_p_f.query('Frequency < 45.25'), 
             x='Frequency', y='Dissimilarity', hue='tinnitus', 
             ax=axes[0], legend=False)
sns.lineplot(data=rdm_ap_f, x='Frequency', y='Dissimilarity', hue='tinnitus', ax=axes[1])
axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[1].set_ylabel('')
plt.tight_layout()

#f.savefig('../results/dissimilarity_metric_euclidean.svg')

#%%
sns.set_context('talk')
f, ax = plt.subplots(figsize=(5, 100))
sns.pointplot(data=rdm_p_s, 
              y='ch_name', 
              x='Dissimilarity', 
              hue='tinnitus', 
              markers='|',
              #errorbar='se',
              dodge=True,
              ax=ax, 
              join=False)
ax.set_xscale('log')

#%%
sns.set_context('talk')
f, ax = plt.subplots(figsize=(5, 100))
sns.pointplot(data=rdm_ap, 
              y='ch_name', 
              x='Dissimilarity', 
              hue='tinnitus', 
              markers='|',
              #errorbar='se',
              dodge=True,
              ax=ax, 
              join=False)
ax.set_xscale('log')



#%%

#rdm_ap
region = "R_FEF_ROI"

sns.pointplot(rdm_ap_s.query(f'ch_name == "{region}"'), x='tinnitus', y='Dissimilarity')

#%%
plt.hist(rdm_ap_s.query(f'ch_name == "{region}"').query('tinnitus == True')['Dissimilarity'])
plt.hist(rdm_ap_sc.query(f'ch_name == "{region}"').query('tinnitus == False')['Dissimilarity'])

#%%
rdm_ap_s.query(f'ch_name == "{region}"').query('tinnitus == True')

#%%


df_cmb = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_all_spec_features.csv')


features =[ 'offset', 'exponent', 'delta_cf', 'theta_cf', 
                    'alpha_cf', 'beta_cf', 'gamma_cf',
                    'n_peaks', 'delta_bw', 'theta_bw', 'alpha_bw',
                    'beta_bw', 'gamma_bw', 'delta_pw', 'theta_pw',
                    'alpha_pw', 'beta_pw', 'gamma_pw', 'delta_osc',
                    'theta_osc', 'alpha_osc', 'beta_osc', 'gamma_osc']
cols_of_interest = ['subject_id', 'ch_name'] + features

# %%
data4rsa = (df_cmb[cols_of_interest]
                 .set_index(['ch_name', 'subject_id'])
                 .to_xarray()
                 .to_array()
                 .to_numpy()
                 .swapaxes(0, 2)
                 )

# %%
n_subjects, n_channels, n_features = data4rsa.shape# each subject is an event
# ch_name x features x subject_ids

# RDM - 1  subjects x features -> aka dissimilarity over features per subject not really
# RDM - 2  features x subjects -> what we want rdm per feature




#
subjects = df_cmb.groupby('subject_id').mean()['tinnitus'].index
tinnitus = df_cmb.groupby('subject_id').mean()['tinnitus'].to_numpy()


data_sets = []
for feat in range(n_features):

    df_cmb['ch_name'].unique()
              
    obs_des = {
              #'features': features,
              'subject': subjects,
              'tinnitus': tinnitus,
               } # observation descriptor --> can be vigorously extended
    des = {#'session': 0,
           'features': features[feat],
           #'subject': subjects[sub],
           #'tinnitus': tinnitus[sub]
           }


    data_sets.append(rsa.data.Dataset(data4rsa[:,:,feat],
                                      descriptors=des,
                                      obs_descriptors=obs_des,
                                      channel_descriptors={'names': df_cmb['ch_name'].unique()}))
# %%
rdms_data_p_feats = rsa.rdm.calc_rdm(data_sets, method='correlation')
# %%
rdms_data_p_feats.dissimilarities
# %%
rsa.vis.rdm_plot.show_rdm(rdms_data_p_feats)
# %%
