#%%
import numpy as np
import rsatoolbox as rsa
import pandas as pd

#%%


def build_rsa_dataset_frequency(data_ct, data_tin, ch_labels, freqs, random=False):

       if random:
              #permutation on the first dimension aka subjects
              data = np.random.permutation(np.concatenate([data_ct, data_tin], axis=1).swapaxes(0, 1))
       else:
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





def build_rsa_dataset_spatial(data_ct, data_tin, ch_labels, freqs, random=False):

       if random:
              #permutation on the first dimension aka subjects
              data = np.random.permutation(np.concatenate([data_ct, data_tin], axis=1).swapaxes(0, 1).swapaxes(1, 2))
       else:
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


def triu2nan(m):
   
   '''Function to turn the upper triangle of a dissimilarity matrix to nans'''
   
   m_new = np.empty(m.shape)
   for freq in range(m.shape[0]):

       m_tmp = m[freq]
       m_tmp[np.triu_indices(m_tmp.shape[1], -1)] = np.nan

       m_new[freq, :, :] = m_tmp
   return m_new



def get_rdm_cross_within(data_sets, freqs, ch_names, distance = 'euclidean', freq=True):
       

       '''Function to extract the difference between across and within group dissimilarities'''

       rdms_data = rsa.rdm.calc_rdm(data_sets, 
                             method=distance, 
                             descriptor='subject'
                             )
       rdm = rdms_data.get_matrices()

       for cur_dim in rdm:
              np.fill_diagonal(cur_dim, np.nan)

       feature_dim = rdm.shape[0]

       #tril2nan is kinda unnecessary but keeping it just incase
       ave_dist_in = np.nanmean(np.concatenate([triu2nan(rdm[:,53:,53:]).reshape(feature_dim, -1), 
                                                triu2nan(rdm[:,:53,:53]).reshape(feature_dim, -1),], axis=1), axis=1)
       ave_dist_cross = np.nanmean(rdm[:,:53, 53:].reshape(feature_dim, -1), axis=1)

       delta_dist = ave_dist_cross - ave_dist_in 


       df_diss = pd.DataFrame(delta_dist)

       if freq:
             df_diss['Frequency'] = freqs
             id_var = 'Frequency'
       else:
             df_diss['ch_name'] = ch_names
             id_var = 'ch_name'
       df_diss.columns = ['Dissimilarity', id_var]

       return df_diss