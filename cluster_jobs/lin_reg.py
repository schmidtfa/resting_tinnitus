#%%
from pathlib import Path
import joblib
import pandas as pd
import pymc as pm
#from pymc.sampling_jax import sample_numpyro_nuts
import numpy as np
from scipy.stats import zscore

from plus_slurm import Job
import arviz as az


# %%
class LinReg(Job):

    def run(self,
            feature,
            ):

            sample_kwargs = {#'progressbar':False,
                            'draws': 2000,
                            'tune': 2000,
                            'chains': 4,
                            'target_accept': 0.99,
                            'nuts_sampler':'numpyro'
                            }

            df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
            df_pe = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params.csv')

            cur_df = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])
            cur_df = cur_df.query('tinnitus == True')
            #%%
            mdf = self._run_lin_reg(cur_df, feature, sample_kwargs)
            ch_effects = az.summary(mdf, var_names='beta|', hdi_prob=.89)

            #%% save
            ch_effects.to_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/lin_reg/{feature}.csv')
            mdf.to_netcdf(f'/home/schmidtfa/experiments/resting_tinnitus/data/lin_reg/{feature}.nc')


     # %% define regression model

    #tinnitus_distress ~ 1 + feature + (1 + feature|channel)

    def _run_lin_reg(self, df, feature, sample_kwargs):

        cur_df = df[[feature, 'tinnitus_distress', 'ch_name']].dropna()

        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        with pm.Model(coords=coords) as glm:
            #Decided to fit an unpooled model. 
            #Partial Pooling over the brain doesnt seem sensible (maybe within roi)
            
            #priors (seperately for hurdle and lognormal)
            alpha = pm.Normal('1|', mu=0, sigma=5, dims="ch_name")
            beta = pm.Normal('beta|', mu=0, sigma=5, dims="ch_name")

            psi_alpha = pm.Normal('psi_1|', mu=0, sigma=1, dims="ch_name")
            psi_beta = pm.Normal('psi_beta|', mu=0, sigma=1, dims="ch_name")

            #likelihood
            sigma = pm.Exponential('sigma',  lam=1)
            #psi = pm.Uniform('psi', 0.1, 0.9)
            observed = pm.HurdleLogNormal('tinnitus_distress',
                                          psi=pm.math.invlogit(psi_alpha[ch_ixs] + psi_beta[ch_ixs]*zscore(cur_df[feature])),
                                          mu=alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature]),
                                          sigma=np.exp(sigma),
                                          observed=cur_df['tinnitus_distress'],
                                          dims="obs_id")

            #mdf = sample_numpyro_nuts(**sample_kwargs)
            mdf =  pm.sample(**sample_kwargs)

        return mdf#, glm

if __name__ == '__main__':

    #%
    feature = 'exponent'
    periodic_type=None#'cf'

    job = LinReg(feature=feature, periodic_type=periodic_type)
    job.run_private()
