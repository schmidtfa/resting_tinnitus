#%%
from pathlib import Path
import joblib
import pandas as pd
import pymc as pm
#from pymc.sampling_jax import sample_numpyro_nuts
import numpy as np
from scipy.stats import zscore

from plus_slurm import Job

# %%
class LogReg(Job):

    def run(self,
            subject_list,
            feature,
            ap_mode,
            periodic_type=None,
            ):

        if np.logical_and(periodic_type != None, feature in ['exponent', 'offset', 'knee_freq']):
            return print('this doesnt make sense')
        else:

            sample_kwargs = {#'progressbar':False,
                            'draws': 2000,
                            'tune': 2000,
                            'chains': 4,
                            'target_accept': 0.95,}

            all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam').glob('*/*[[]0.5, 100[]].dat'))

            # %%
            periodic, aperiodic = [], []

            for f in all_files:
                
                cur_data = joblib.load(f)

                periodic.append(cur_data['periodic'])
                aperiodic.append(cur_data['aperiodic'])

            # %% 

            df_periodic = pd.concat(periodic).query('subject_id == @subject_list')
            df_aperiodic = pd.concat(aperiodic).query('subject_id == @subject_list')
            #%% Test for physiological differences in aperiodic activity (tinnitus vs. control)
            physio = ['ECG', 'EOGV', 'EOGH']

            if feature in ['exponent', 'offset', 'knee_freq']:
                cur_df = (df_aperiodic.query('ch_name != @physio')
                                    .query(f'aperiodic_mode == "{ap_mode}"'))

            elif feature in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                cur_df = (df_periodic.query('ch_name != @physio')
                                    .query(f'aperiodic_mode == "{ap_mode}"')
                                    .query(f'peak_params == "{periodic_type}"'))

            #%%
            mdf = self._run_log_reg(cur_df, feature, sample_kwargs)

            #%% save
            if feature in ['exponent', 'offset', 'knee_freq']:
                mdf.to_netcdf(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/log_reg/{feature}_{ap_mode}.nc')
            elif feature in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                mdf.to_netcdf(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/log_reg/{feature}_{ap_mode}_{periodic_type}.nc')


     # %% define regression model
    def _run_log_reg(self, df, feature, sample_kwargs, non_centered=True):

        cur_df = df[[feature, 'tinnitus', 'ch_name']].dropna()

        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        with pm.Model(coords=coords) as glm:


            if non_centered:
                mu_a = pm.Normal('intercept', 0, 1)
                z_a = pm.Normal('z_a', 0, 1, dims="ch_name")
                sigma_a = pm.Exponential('sigma_intercept', lam=1)


                mu_b = pm.Normal('beta', 0, .5)
                z_b = pm.Normal('z_b', 0, .5, dims="ch_name")
                sigma_b = pm.Exponential('sigma_beta', lam=1)

                #channel priors centered parametrization -> surprisingly faster than non-centered
                alpha = pm.Deterministic('1|', mu_a + z_a * sigma_a, dims="ch_name")
                beta = pm.Deterministic('beta|', mu_b + z_b * sigma_b, dims="ch_name")
            
            else:
                #Hyperpriors
                a = pm.Normal('intercept', 0, 1.5)
                sigma_a = pm.Exponential('sigma_intercept', lam=1)
                b = pm.Normal('beta', 0, 1)
                sigma_b = pm.Exponential('sigma_beta', lam=1)

                #channel priors centered parametrization -> surprisingly faster than non-centered
                alpha = pm.Normal('1|', mu=a, sigma=sigma_a, dims="ch_name")
                beta = pm.Normal('beta|', mu=b, sigma=sigma_b, dims="ch_name")

            #likelihood
            observed = pm.Bernoulli('tinnitus',
                                    p=pm.math.invlogit(alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature])),
                                    observed=cur_df['tinnitus'],
                                    dims="obs_id")

            #mdf = sample_numpyro_nuts(**sample_kwargs)
            mdf =  pm.sample(**sample_kwargs)

        return mdf#, glm

if __name__ == '__main__':

    #%
    feature = 'exponent'
    ap_mode = 'knee'
    periodic_type=None#'cf'

    job = LogReg(feature=feature, ap_mode=ap_mode, periodic_type=periodic_type)
    job.run_private()