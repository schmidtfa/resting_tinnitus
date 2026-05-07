#%%

import pandas as pd
import pymc as pm
import numpy as np


from plus_slurm import Job
import arviz as az
import pytensor.tensor as pt
from pathlib import Path

# %%
class AgeReg(Job):

    def run(self,
            cortex=True,
            ):

            #%%
            sample_kwargs = {#'progressbar':False,
                            'draws': 4000,
                            'tune': 4000,
                            'chains': 4,
                            #'cores': 1,
                            'target_accept': 0.99,
                            'nuts_sampler':"blackjax"
                            #'nuts_sampler':'numpyro',
                            #'idata_kwargs': {"log_likelihood": True}
                            }

            #feature = 'Exponent_2'
            base_path = Path('/home/schmidtfa/experiments/brain_age/data/data_cam_can')
            if cortex:
               df_gm = pd.read_csv(base_path / 'gm_volume.csv')
               #% merge with cortex labels and pick cur label for pooling
               df_regions_info = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/regions_hcmp.csv')
               df_regions_info['roi'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
               df_regions_info['roi'] = df_regions_info['roi'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                                        'R_7Pl_ROI': 'R_7PL_ROI',})

               df_regions_info['cortex_info'] = df_regions_info['roi'] + '_' + df_regions_info['LR']

               df_cmb = df_regions_info.merge(df_gm, on='roi') #This order is fucking important!!! TODO: check every merge
               #TODO: Think of a better informed way of dropping duplicate oscillations e.g. pick highest per sub and channel amplitude
               #df_cmb.drop_duplicates(subset=['subject_id', 'ch_name'], inplace=True)

            else:
               df_cmb = pd.read_csv(base_path / 'aseg_volume.csv')
               df_cmb.rename(columns={'vol_norm': 'volume_norm',
                                      'StructName': 'roi'}, inplace=True)
            
            feature = 'volume_norm' #normalized by tiv

            print('Running model')

            #%%
            if cortex:
               self._run_hierarchical_reg(df_cmb, 
                                             feature,
                                             df_regions_info, 
                                             sample_kwargs)
               
            else:
               self._run_reg(df_cmb, 
                              feature,
                              sample_kwargs)
    # %% define regression model
    def _run_reg(self,
                 df,
                 feature,
                 sample_kwargs):
        
        cur_df = df[['roi', feature, 'subject', 'age']].dropna()


        standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

        #consider areas with 0 volume NAN
        cur_df[feature][cur_df[feature] == 0] = np.nan
        cur_df = cur_df[np.isnan(cur_df[feature]) == False].copy()
        cur_df[feature] = np.log(cur_df[feature]) #distribution is very heavy tailed

        cur_df['age_centered'] = (cur_df['age'] - cur_df['age'].mean()) / 10
        cur_df['age_z'] = (
               cur_df.groupby("roi")['age']
                      .transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x)))
        )
        cur_df[feature] = (
               cur_df.groupby("roi")[feature]
                      .transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x)))
        )
        
        ch_ixs, channel = pd.factorize(cur_df['roi'])
        coords = {
        "roi": channel,
        "obs_id": np.arange(len(ch_ixs)),
        }

        with pm.Model(coords=coords) as glm:

            age   = pm.Data('age', cur_df['age_z'].values, dims='obs_id')

            y_s   = pm.Data('y_s', cur_df[feature].values, dims='obs_id')
            ch_ix = pm.Data('ch_ix', ch_ixs, dims='obs_id')


            alpha = pm.StudentT('alpha', nu=10, mu=0, sigma=1, dims='roi')
            beta = pm.Normal('beta_ch', mu=0, sigma=.5, dims='roi')

            # linear model
            sigma = pm.Exponential('sigma', lam=1)
            nu = pm.Gamma('nu', 2, 0.1)
            y = pm.StudentT('y',
                            mu=alpha[ch_ix] + beta[ch_ix] * age,
                            observed=y_s,
                            nu=nu,
                            sigma=sigma,
                            dims='obs_id'
                            ) 
        
            mdf =  pm.sample(**sample_kwargs)

        ch_effects = az.summary(mdf, 
                    var_names='beta_ch', 
                    hdi_prob=.89)

        ch_effects.to_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/age_cc_final_final/gm_aseg_{feature}.csv')
        mdf.to_netcdf(f'/home/schmidtfa/experiments/resting_tinnitus/data/age_cc_final_final/gm_aseg_{feature}.nc')


    def _run_hierarchical_reg(self, 
                                  df, 
                                  feature, 
                                  df_regions_info, 
                                  sample_kwargs):

        #feature = 'tau'
        cur_df = df[['cortex_info', 'roi', feature, 'subject', 'age']].dropna()


        #standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

        #consider areas with 0 volume NAN
        cur_df[feature][cur_df[feature] == 0] = np.nan
        cur_df = cur_df[np.isnan(cur_df[feature]) == False].copy()
        cur_df[feature] = np.log(cur_df[feature]) #distribution is very heavy tailed

        cur_df['age_centered'] = (cur_df['age'] - cur_df['age'].mean()) / 10
        cur_df['age_z'] = (
               cur_df.groupby("roi")['age']
                      .transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x)))
        )

        cur_df[feature] = (
               cur_df.groupby("roi")[feature]
                      .transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x)))
        )

        #4debug
        ch_ixs, channel = pd.factorize(cur_df['roi'])
        coords = {
        "roi": channel,
        "obs_id": np.arange(len(ch_ixs)),
        }

        ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
        coords['cortices'] = cortices


        with pm.Model(coords=coords) as glm:

            age   = pm.Data('age', cur_df['age_z'].values, dims='obs_id')

            y_s   = pm.Data('y_s', cur_df[feature].values, dims='obs_id')
            ch_ix = pm.Data('ch_ix', ch_ixs, dims='obs_id')

            #l1 cortices intercepts (loose)
            # alpha_m1 = pm.StudentT('alpha_m1', nu=5, mu=0, sigma=2.5, dims='cortices')
            # alpha_s1 = pm.Gamma('alpha_s1', 2, .5, dims='cortices')

            #l1 cortices intercepts (tighter)
            alpha_m1 = pm.StudentT('alpha_m1', nu=10, mu=0, sigma=1, dims='cortices')
            alpha_s1 = pm.HalfNormal('alpha_s1', sigma=.5, dims='cortices')

            #l1 cortices slopes (loose)
            # beta_m1 = pm.StudentT('beta_m1', nu=5, mu=0, sigma=2.5, dims=('cortices', 'effect'))
            # beta_s1 = pm.Gamma('beta_s1', 2, .5, dims=('cortices', 'effect'))

            #l1 cortices slopes (tighter)
            beta_m1 = pm.Normal('beta_m1', mu=0, sigma=.5, dims='cortices')
            beta_s1 = pm.HalfNormal('beta_s1', sigma=.5, dims='cortices',)

            #l2 parcels nested in cortices intercepts (loose)
            #alpha_m2 = pm.StudentT('alpha_m2', nu=5, mu=0, sigma=2.5, dims='ch_name')

            #l2 parcels nested in cortices intercepts (tight)
            alpha_m2 = pm.StudentT('alpha_m2', nu=10, mu=0, sigma=1, dims='roi')

            #l2 parcels nested in cortices slopes (loose)
            #beta_m2 = pm.StudentT('beta_m2', nu=5, mu=0, sigma=2.5, dims=('ch_name', 'effect'))

            #l2 parcels nested in cortices slopes (tight)
            beta_m2 = pm.Normal('beta_m2', mu=0, sigma=.5, dims='roi')

            #non-centered parametrization cortex + roi
            alpha_ch = pm.Deterministic(
                 'alpha_ch', 
                  alpha_m1[ch2cort_ixs] + alpha_m2 * alpha_s1[ch2cort_ixs], 
                  dims='roi')
            beta_ch = pm.Deterministic(
                 'beta_ch', 
                  beta_m1[ch2cort_ixs] + beta_m2 * beta_s1[ch2cort_ixs], 
                  dims='roi')

            # linear model
            sigma = pm.Exponential('sigma', lam=1)
            nu = pm.Gamma('nu', 2, 0.1)
            y = pm.StudentT('y',
                            mu=alpha_ch[ch_ix] + beta_ch[ch_ix] * age,
                            observed=y_s,
                            nu=nu,
                            sigma=sigma,
                            dims='obs_id'
                            ) 

            mdf =  pm.sample(**sample_kwargs)

        ch_effects = az.summary(mdf, 
                    var_names='beta_ch', 
                    hdi_prob=.89)

        ch_effects.to_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/age_cc_final_final/gm_{feature}_hi.csv')
        mdf.to_netcdf(f'/home/schmidtfa/experiments/resting_tinnitus/data/age_cc_final_final/gm_{feature}_hi.nc')

    
