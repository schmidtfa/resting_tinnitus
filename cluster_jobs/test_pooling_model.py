#%%
import pymc as pm
import arviz as az

import pandas as pd

import numpy as np

from scipy.stats import zscore

from plus_slurm import Job
from collections.abc import Iterable
from pathlib import Path

import os

import sys
sys.path.append('/home/schmidtfa/experiments/resting_tinnitus/utils')

from pooling_sim_utils import pooling_sim

# %%
class SimPool(Job):

    def run(self,
            cortex_of_interest = "Primary_Visual",
            slope_effect = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            cur_seed = 42,
            n_subjects = 40,
    ):


        if isinstance(slope_effect, Iterable):
            outdir='slope_varying'
        else:
            outdir='slope_fixed'
        #%%

        sample_kwargs = {#'progressbar':False,
                    'draws': 4000,
                    'tune': 4000,
                    'chains': 4,
                    'cores': 1,
                    'target_accept': 0.95,
                    'nuts_sampler':"blackjax"
                    #'nuts_sampler':'numpyro',
                    #'idata_kwargs': {"log_likelihood": True}
                    }
        df_regions_info = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/regions_hcmp.csv')
        df_regions_info['roi'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
        df_regions_info['roi'] = df_regions_info['roi'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                                 'R_7Pl_ROI': 'R_7PL_ROI',})

        df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']

        #%%

        df_roi = df_regions_info[['roi', 'cortex_info']]

        df_all, eff_roi, df_sim_params = pooling_sim(df_roi,
                                        cortex_of_interest=cortex_of_interest,
                                        intercept = 0,
                                        slope_effect = slope_effect,
                                        slope_null_effect = 0.,
                                        seed_value=cur_seed,
                                        n_subjects = n_subjects)

        #%%

        ch_ixs, channel = pd.factorize(df_all['roi'])
        coords = {
        "roi": channel,
        "obs_id": np.arange(len(ch_ixs)),
        }

        ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
        coords['cortices'] = cortices

        with pm.Model(coords=coords) as glme:

            X_s   = pm.Data('X_s', df_all['X'].values, dims='obs_id')
            y_s   = pm.Data('y_s', df_all['y'].values, dims='obs_id')

            ch_ix = pm.Data('ch_ix', ch_ixs, dims='obs_id')

            #l1 cortices intercepts (tighter)
            alpha_m1 = pm.StudentT('alpha_m1', nu=10, mu=0, sigma=1, dims='cortices')
            alpha_s1 = pm.HalfNormal('alpha_s1', sigma=.5, dims='cortices')

            #l1 cortices slopes (tighter)
            beta_m1 = pm.Normal('beta_m1', mu=0, sigma=.5, dims='cortices')
            beta_s1 = pm.HalfNormal('beta_s1', sigma=.5, dims='cortices')

            #l2 parcels nested in cortices intercepts (tight)
            alpha_m2 = pm.StudentT('alpha_m2', nu=10, mu=0, sigma=1, dims='roi')

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
                            mu=alpha_ch[ch_ix] + beta_ch[ch_ix] * X_s,
                            observed=y_s,
                            nu=nu,
                            sigma=sigma,
                            dims='obs_id'
                            ) 


        #%% unpooled model
        ch_ixs, channel = pd.factorize(df_all['roi'])
        coords = {
        "roi": channel,
        "obs_id": np.arange(len(ch_ixs)),
        }

        with pm.Model(coords=coords) as glm:

            X_s   = pm.Data('X_s', df_all['X'].values, dims='obs_id')
            y_s   = pm.Data('y_s', df_all['y'].values, dims='obs_id')

            ch_ix = pm.Data('ch_ix', ch_ixs, dims='obs_id')

            alpha = pm.StudentT('alpha', nu=10, mu=0, sigma=1, dims='roi')
            beta = pm.Normal('beta', mu=0, sigma=.5, dims='roi')

            # linear model
            sigma = pm.Exponential('sigma', lam=1)
            nu = pm.Gamma('nu', 2, 0.1)
            y = pm.StudentT('y',
                            mu=alpha[ch_ix] + beta[ch_ix] * X_s,
                            observed=y_s,
                            nu=nu,
                            sigma=sigma,
                            dims='obs_id'
                            ) 
            

        #%% lme full classic model
        ch_ixs, channel = pd.factorize(df_all['roi'])
        coords = {
        "roi": channel,
        "obs_id": np.arange(len(ch_ixs)),
        }

        with pm.Model(coords=coords) as glme_classic:

            X_s   = pm.Data('X_s', df_all['X'].values, dims='obs_id')
            y_s   = pm.Data('y_s', df_all['y'].values, dims='obs_id')

            ch_ix = pm.Data('ch_ix', ch_ixs, dims='obs_id')

            alpha_m1 = pm.StudentT('alpha_m1', nu=10, mu=0, sigma=1)
            alpha_s1 = pm.HalfNormal('alpha_s1', sigma=.5)

            #l1 cortices slopes (tighter)
            beta_m1 = pm.Normal('beta_m1', mu=0, sigma=.5)
            beta_s1 = pm.HalfNormal('beta_s1', sigma=.5)

            #l2 parcels nested in cortices intercepts (tight)
            alpha_m2 = pm.StudentT('alpha_m2', nu=10, mu=0, sigma=1, dims='roi')

            #l2 parcels nested in cortices slopes (tight)
            beta_m2 = pm.Normal('beta_m2', mu=0, sigma=.5, dims='roi')

            #non-centered parametrization cortex + roi
            alpha_ch = pm.Deterministic(
                    'alpha_ch', 
                    alpha_m1 + alpha_m2 * alpha_s1, 
                    dims='roi')
            beta_ch = pm.Deterministic(
                    'beta_ch', 
                    beta_m1 + beta_m2 * beta_s1, 
                    dims='roi')

            # linear model
            sigma = pm.Exponential('sigma', lam=1)
            nu = pm.Gamma('nu', 2, 0.1)
            y = pm.StudentT('y',
                            mu=alpha_ch[ch_ix] + beta_ch[ch_ix] * X_s,
                            observed=y_s,
                            nu=nu,
                            sigma=sigma,
                            dims='obs_id'
                            ) 

        #%%
        with glm:
            mdf_lm = pm.sample(**sample_kwargs)

        #%%
        with glme:
            mdf_lme = pm.sample(**sample_kwargs)

        #%%
        with glme_classic:
            mdf_lme_class = pm.sample(**sample_kwargs)

        #%%

        summary_lm = az.summary(mdf_lm, hdi_prob=0.89).reset_index()
        summary_lme = az.summary(mdf_lme, hdi_prob=0.89).reset_index()
        summary_lme_c = az.summary(mdf_lme_class, hdi_prob=0.89).reset_index()

        # %%
        channel_effects_lme = summary_lme.iloc[[True if "beta_ch" in i else False for i in summary_lme['index']]]
        channel_effects_lme['roi'] = [i[8:-1] for i in channel_effects_lme['index']]

        #%%
        channel_effects_lme_c = summary_lme_c.iloc[[True if "beta_ch" in i else False for i in summary_lme_c['index']]]
        channel_effects_lme_c['roi'] = [i[8:-1] for i in channel_effects_lme_c['index']]

        #%%
        channel_effects_lm = summary_lm.iloc[[True if "beta" in i else False for i in summary_lm['index']]]
        channel_effects_lm['roi'] = [i[5:-1] for i in channel_effects_lm['index']]

        # %%
        def get_effect(df, df_sim_params, model_type):
            effect_mask = np.logical_or(df['hdi_5.5%'] > 0.05, df['hdi_94.5%'] < -0.05)

            df['mean_mask'] = df['mean'] * effect_mask

            df = df.merge(df_sim_params, on='roi')
            df['model'] = model_type

            return df[['mean', 'mean_mask', 'sd', 'hdi_5.5%', 'hdi_94.5%', 'sim_slopes', 'roi', 'model']].copy()



        df_lm = get_effect(channel_effects_lm, df_sim_params, model_type='unpooled')
        df_lme = get_effect(channel_effects_lme, df_sim_params, model_type='lme_cortical')
        df_lme_c = get_effect(channel_effects_lme_c, df_sim_params, model_type='lme_classic')


        df_eff = pd.concat([df_lm, df_lme, df_lme_c])
        df_eff['cur_seed'] = cur_seed
        df_eff['effs_simulated'] = len(eff_roi)
        # %%

        df_sim = pd.DataFrame({'effs_simulated': len(eff_roi),
                    'lme_correctly_detected': df_lme[df_lme['mean_mask'] != 0].query('roi == @eff_roi').shape[0],
                    'lme_classic_correctly_detected': df_lme_c[df_lme_c['mean_mask'] != 0].query('roi == @eff_roi').shape[0],
                    'lm_unpooled_correctly_detected': df_lm[df_lm['mean_mask'] != 0].query('roi == @eff_roi').shape[0],
                    'lme_incorrectly_detected': df_lme[df_lme['mean_mask'] != 0].query('roi != @eff_roi').shape[0],
                    'lme_classic_incorrectly_detected': df_lme_c[df_lme_c['mean_mask'] != 0].query('roi != @eff_roi').shape[0],
                    'lm_unpooled_incorrectly_detected': df_lm[df_lm['mean_mask'] != 0].query('roi != @eff_roi').shape[0],
                    'location': cortex_of_interest,
                    }, index=[0])
        df_sim['cur_seed'] = cur_seed

        out_p = Path('/home/schmidtfa/experiments/resting_tinnitus/data/sim_pool/') / outdir

        if not os.path.isdir(out_p):
            os.makedirs(out_p)
            
        df_sim.to_csv(out_p / f'sim_{cortex_of_interest}_{cur_seed}.csv')
        df_eff.to_csv(out_p / f'eff_{cortex_of_interest}_{cur_seed}.csv')