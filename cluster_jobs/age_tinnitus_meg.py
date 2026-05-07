#%%

import pandas as pd
import pymc as pm
import numpy as np


from plus_slurm import Job
import arviz as az
import pytensor.tensor as pt

# %%
class AgeReg(Job):

    def run(self,
            feature,
            thresh=3.0
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

               #feature = 'alpha_pw'
               df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
               df_pe = pd.read_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params__peak_threshold_{thresh}.csv')

               cur_df_f = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])

               #%% merge with cortex labels and pick cur label for pooling
               df_regions_info = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/regions_hcmp.csv')
               df_regions_info['ch_name'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
               df_regions_info['ch_name'] = df_regions_info['ch_name'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                                                'R_7Pl_ROI': 'R_7PL_ROI',})

               df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']

               df_cmb = df_regions_info.merge(cur_df_f, on='ch_name') #This order is fucking important!!! TODO: check every merge
               #TODO: Think of a better informed way of dropping duplicate oscillations e.g. pick highest per sub and channel amplitude
               #df_cmb.drop_duplicates(subset=['subject_id', 'ch_name'], inplace=True)

               print('Running model')


               #feature = 'alpha_cf'
               cur_df = df_cmb[['cortex_info', 'ch_name', feature, 'tinnitus', 'subject_id', 'age']].dropna()


               standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

               if feature in ['Exponent_1', 'tau']:
                    cur_df[feature] = np.log(cur_df[feature])
                    
               if feature == 'alpha_pw':
                    cur_df = cur_df[np.isnan(cur_df[feature]) == False].copy()
                    cur_df[feature] = np.log(cur_df[feature])

                    #drop columns with 0 variance before zscoring
               elif feature in ['alpha_cf', 'alpha_bw']:
                    cur_df = cur_df[np.isnan(cur_df[feature]) == False].copy() 

               #%%
               cur_df['age_centered'] = (cur_df['age'] - cur_df['age'].mean()) / 10
               #standardize
               cur_df['age_z'] = (
                                 cur_df.groupby("ch_name")['age']
                                       .transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x)))
               )

               cur_df[feature] = (cur_df.groupby("ch_name")[feature]
                                        .transform(lambda x: (x - np.nanmean(x)) / (np.nanstd(x)))
                                   )
               #effect coding for tinnitus to reduce multicollinearity
               cur_df['tinnitus'] = cur_df['tinnitus'] - .5

               #4debug
               ch_ixs, channel = pd.factorize(cur_df['ch_name'])
               coords = {
               "ch_name": channel,
               "obs_id": np.arange(len(ch_ixs)),
               }

               ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
               coords['cortices'] = cortices
               coords['effect'] = ["age", "tinnitus", "age:tinnitus"]


               with pm.Model(coords=coords) as glm:

                    age   = pm.Data('age', cur_df['age_z'].values, dims='obs_id')
                    tinnitus = pm.Data('tinnitus', cur_df['tinnitus'].values, dims='obs_id')

                    y_s   = pm.Data('y_s', cur_df[feature].values, dims='obs_id')
                    ch_ix = pm.Data('ch_ix', ch_ixs, dims='obs_id')

                    # Design matrix: [age, tinnitus, age * tinnitus]
                    X = pt.stack([age, tinnitus, age * tinnitus], axis=1)

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
                    beta_m1 = pm.Normal('beta_m1', mu=0, sigma=.5, dims=('cortices', 'effect'))
                    beta_s1 = pm.HalfNormal('beta_s1', sigma=.5, dims=('cortices', 'effect'))

                    #l2 parcels nested in cortices intercepts (loose)
                    #alpha_m2 = pm.StudentT('alpha_m2', nu=5, mu=0, sigma=2.5, dims='ch_name')

                    #l2 parcels nested in cortices intercepts (tight)
                    alpha_m2 = pm.StudentT('alpha_m2', nu=10, mu=0, sigma=1, dims='ch_name')

                    #l2 parcels nested in cortices slopes (loose)
                    #beta_m2 = pm.StudentT('beta_m2', nu=5, mu=0, sigma=2.5, dims=('ch_name', 'effect'))

                    #l2 parcels nested in cortices slopes (tight)
                    beta_m2 = pm.Normal('beta_m2', mu=0, sigma=.5, dims=('ch_name', 'effect'))

                    #non-centered parametrization cortex + roi
                    alpha_ch = pm.Deterministic(
                         'alpha_ch', 
                         alpha_m1[ch2cort_ixs] + alpha_m2 * alpha_s1[ch2cort_ixs], 
                         dims='ch_name')
                    beta_ch = pm.Deterministic(
                         'beta_ch', 
                         beta_m1[ch2cort_ixs, :] + beta_m2 * beta_s1[ch2cort_ixs, :], 
                         dims=('ch_name', 'effect'))

                    # linear model
                    sigma = pm.Exponential('sigma', lam=1)
                    nu = pm.Gamma('nu', 2, 0.1)
                    y = pm.StudentT('y',
                                   mu=alpha_ch[ch_ix] + pt.batched_dot(beta_ch[ch_ix], X),
                                   observed=y_s,
                                   nu=nu,
                                   sigma=sigma,
                                   dims='obs_id'
                                   ) 

                    mdf =  pm.sample(**sample_kwargs)

               ch_effects = az.summary(mdf, 
                              var_names='beta_ch', 
                              hdi_prob=.89)

               ch_effects.to_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/age_tinn_final_final/{feature}_hi__peak_threshold_{thresh}.csv')
               mdf.to_netcdf(f'/home/schmidtfa/experiments/resting_tinnitus/data/age_tinn_final_final/{feature}_hi__peak_threshold_{thresh}.nc')

    
