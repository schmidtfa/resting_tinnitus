#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import joblib
import numpy as np
import rsatoolbox as rsa



import scipy.stats as st 
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')

# %%
#%%
df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = sorted(df_all['subject_id'].unique())
#__fft_method_multitaper.dat
#bad subjects


#%%

spectra_dfs_chs = []

for subject_id in subject_ids:

    INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
    cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__src_type_beamformer.dat'))[0])

    freq_cut = np.logical_and(cur_data['freq'] > .2, cur_data['freq'] < 98.25)

    cur_df = pd.DataFrame(cur_data['src']['label_tc'].T[freq_cut], 
                       columns=cur_data['src']['label_info']['names_order_mne'],
                       index=cur_data['freq'][freq_cut])
    cur_df_tidy = cur_df.reset_index().melt(id_vars='index',
                                            var_name='ch_name',
                                            value_name='Power (a.u.)')
    cur_df_tidy.rename(columns={'index' :'Frequency (Hz)'}, inplace=True)
    cur_df_tidy['subject_id'] = subject_id
    
    spectra_dfs_chs.append(cur_df_tidy)
# %%
feature_list = ['ch_name', 'subject_id', 'Frequency (Hz)', 'Power (a.u.)']

raw_spectra_no_t = np.squeeze(pd.concat(spectra_dfs_chs)
                 .query('ch_name != "???"')
                 .merge(df_all[['subject_id', 'tinnitus']], on='subject_id')
                 .query('tinnitus == False')[feature_list]
                 .set_index(['ch_name', 'subject_id', 'Frequency (Hz)'])
                 .to_xarray()
                 .to_array()
                 .to_numpy())

raw_spectra_t = np.squeeze(pd.concat(spectra_dfs_chs)
                 .query('ch_name != "???"')
                 .merge(df_all[['subject_id', 'tinnitus']], on='subject_id')
                 .query('tinnitus == True')[feature_list]
                 .set_index(['ch_name', 'subject_id', 'Frequency (Hz)'])
                 .to_xarray()
                 .to_array()
                 .to_numpy())
# %%
ch_labels = cur_data['src']['label_info']['names_order_mne'][2:]
freqs = np.arange(0.25, 98.25, 0.25)


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

# %%
data_sets = build_rsa_dataset(raw_spectra_no_t, raw_spectra_t)
data_sets_spatial = build_rsa_dataset_spatial(raw_spectra_no_t, raw_spectra_t)


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


       ave_dist_in = np.nanmean(tril2nan((rdm[:,53:,53:] + rdm[:,:53,:53]) /2), axis=2)
       ave_dist_cross = np.nanmean(tril2nan(rdm[:,:53, 53:]), axis=2)


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


#%%
rdm_f = get_rdm(data_sets, freqs=freqs, ch_names=ch_labels, freq=True)
rdm_s = get_rdm(data_sets_spatial, freqs=freqs, ch_names=ch_labels, freq=False)


#%%
sns.set_context('poster')
f, ax = plt.subplots(figsize=(4, 4))

sns.lineplot(data=rdm_f, x='Frequency', y='Dissimilarity', hue='tinnitus', ax=ax)
ax.set_yscale('log')
ax.set_xscale('log')

#plt.tight_layout()
rdm_f.to_csv('../results/raw_rsa_tinnitus_freq.csv')
f.savefig('../results/raw_rsa_tinnitus.svg')

#%%
sns.set_context('talk')
f, ax = plt.subplots(figsize=(5, 100))
sns.pointplot(data=rdm_s, 
              y='ch_name', 
              x='Dissimilarity', 
              hue='tinnitus', 
              markers='|',
              errorbar='se',
              dodge=True,
              ax=ax, 
              join=False)
ax.set_xscale('log')

# %%
