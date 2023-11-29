#%%
from pathlib import Path
import joblib
import pandas as pd
import pymc as pm
from pymc.sampling_jax import sample_numpyro_nuts
import numpy as np
from scipy.stats import zscore

from plus_slurm import Job
import arviz as az
from scipy.spatial.distance import squareform, pdist

# %%
class LogReg(Job):

    def run(self,
            feature,
            ):

            sample_kwargs = {#'progressbar':False,
                            'draws': 2000,
                            'tune': 2000,
                            'chains': 4,
                            'target_accept': 0.99,}

            #feature = 'exponent'
            cur_df_f = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_all_spec_features.csv')
            #%% merge with cortex labels and pick cur label for pooling
            df_regions_info = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/regions_hcmp.csv')
            df_regions_info['ch_name'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
            df_regions_info['ch_name'] = df_regions_info['ch_name'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                                             'R_7Pl_ROI': 'R_7PL_ROI',})

            df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']

            df_cmb = df_regions_info.merge(cur_df_f, on='ch_name') #This order is fucking important!!! TODO: check every merge


            print('Running model')

            mdf_hi = self._run_hierarchical_log_reg(df_cmb, feature, df_regions_info, sample_kwargs)
            ch_effects_hi = az.summary(mdf_hi, 
                                    var_names='beta_ch', 
                                    hdi_prob=.89)
            
            mdf_up = self._run_unpooled_log_reg(df_cmb, feature, df_regions_info, sample_kwargs)
            ch_effects_up = az.summary(mdf_up, 
                                    var_names='beta_ch', 
                                    hdi_prob=.89)

            #%% save
            ch_effects_hi.to_csv(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/log_reg_final/{feature}_hi.csv')
            mdf_hi.to_netcdf(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/log_reg_final/{feature}_hi.nc')

            ch_effects_up.to_csv(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/log_reg_final/{feature}_up.csv')
            mdf_up.to_netcdf(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/log_reg_final/{feature}_up.nc')


    # %% define regression model
    def _run_hierarchical_log_reg(self, df, feature, df_regions_info, sample_kwargs):

        cur_df = df[['cortex_info', 'ch_name', feature, 'tinnitus', 'subject_id']].dropna()

        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        #df_dist = df_regions_info.copy()[['ch_name', 'x-cog', 'y-cog', 'z-cog']]
        #reindex_array_ch = [np.argmax(ch == channel) for ch in df_dist['ch_name']]
        #distance_matrix = squareform(pdist(df_dist.iloc[reindex_array_ch][['x-cog', 'y-cog', 'z-cog']])) / 10 # in cm is easier to sample

        standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

        ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
        coords['cortices'] = cortices
    
        with pm.Model(coords=coords) as glm:

            #l1 cortices
            alpha_m1 = pm.StudentT('alpha_m1', nu=5, mu=0, sigma=2.5, dims='cortices')
            beta_m1 = pm.StudentT('beta_m1', nu=5, mu=0, sigma=2.5, dims='cortices')

            alpha_s1 = pm.Gamma('alpha_s1', 2, .5, dims='cortices')
            beta_s1 = pm.Gamma('beta_s1', 2, .5, dims='cortices')

            #l2 parcels nested in cortices
            alpha_m2 = pm.StudentT('alpha_m2', nu=5, mu=0, sigma=2.5, dims='ch_name')
            beta_m2 = pm.StudentT('beta_m2', nu=5, mu=0, sigma=2.5, dims='ch_name')

            alpha_ch = pm.Deterministic('alpha_ch', alpha_m1[ch2cort_ixs] + alpha_m2 * alpha_s1[ch2cort_ixs], dims='ch_name')
            beta_ch = pm.Deterministic('beta_ch', beta_m1[ch2cort_ixs] + beta_m2 * beta_s1[ch2cort_ixs], dims='ch_name')

            # #likelihood
            y = pm.Bernoulli('y',
                             p=pm.math.invlogit(alpha_ch[ch_ixs] + beta_ch[ch_ixs]*standardize(cur_df[feature])),
                             observed=cur_df['tinnitus'],
                             dims='obs_id'
                             ) 

            mdf = sample_numpyro_nuts(**sample_kwargs)
            #mdf =  pm.sample(**sample_kwargs)


        return mdf#, glm
    


    def _run_unpooled_log_reg(self, df, feature, df_regions_info, sample_kwargs):

        cur_df = df[['cortex_info', 'ch_name', feature, 'tinnitus', 'subject_id']].dropna()

        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        #df_dist = df_regions_info.copy()[['ch_name', 'x-cog', 'y-cog', 'z-cog']]
        #reindex_array_ch = [np.argmax(ch == channel) for ch in df_dist['ch_name']]
        #distance_matrix = squareform(pdist(df_dist.iloc[reindex_array_ch][['x-cog', 'y-cog', 'z-cog']])) / 10 # in cm is easier to sample

        standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

        ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
        coords['cortices'] = cortices
    
        with pm.Model(coords=coords) as glm:

            #l2 parcels
            alpha_ch = pm.StudentT('alpha_ch', nu=5, mu=0, sigma=2.5, dims='ch_name')
            beta_ch = pm.StudentT('beta_ch', nu=5, mu=0, sigma=2.5, dims='ch_name')

            # #likelihood
            y = pm.Bernoulli('y',
                            p=pm.math.invlogit(alpha_ch[ch_ixs] + beta_ch[ch_ixs]*standardize(cur_df[feature])),
                            observed=cur_df['tinnitus'],
                            dims='obs_id'
                            ) 

            mdf = sample_numpyro_nuts(**sample_kwargs)
            #mdf =  pm.sample(**sample_kwargs)


        return mdf#, glm

if __name__ == '__main__':

    #%
    feature = 'exponent'


    job = LogReg(feature=feature)
    job.run_private()



