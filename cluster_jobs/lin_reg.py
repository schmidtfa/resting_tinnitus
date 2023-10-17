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
            subject_list,
            feature,
            low_freq=0.25,
            up_freq=98,
            mask_level=.90,
            periodic_type=None,
            ):

        if np.logical_and(periodic_type != None, feature in ['exponent', 'offset', 'knee_freq', 'n_peaks']):
            return print('this doesnt make sense')
        elif np.logical_and(periodic_type == None, feature in ['delta', 'theta', 'alpha', 'beta', 'gamma']):
            return print('this also makes no sense')
        
        else:

            sample_kwargs = {#'progressbar':False,
                            'draws': 2000,
                            'tune': 2000,
                            'chains': 4,
                            'target_accept': 0.95,}

            all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam').glob(f'*/*__peak_threshold_2.5__freq_range_[[]{low_freq}, {up_freq}[]].dat'))
            #%%
            periodic, aperiodic = [], []

            for f in all_files:
                
                cur_data = joblib.load(f)

                periodic.append(cur_data['periodic'])
                aperiodic.append(cur_data['aperiodic'])

            #%% 
            df_periodic = pd.concat(periodic).query('subject_id == @subject_list')
            df_aperiodic = pd.concat(aperiodic).query('subject_id == @subject_list')
            #%% Test for physiological differences in aperiodic activity (tinnitus vs. control)
            #%% Test for physiological differences in aperiodic activity (tinnitus vs. control)
            physio = ['ECG', 'EOGV', 'EOGH']

            if feature in ['exponent', 'offset', 'knee_freq']:
                cur_df = (df_aperiodic.query('ch_name != @physio')
                                      .query('tinnitus == True')
                                     )

            elif feature in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'n_peaks']:
                if feature != 'n_peaks':
                    cur_df = (df_periodic.query('ch_name != @physio')
                                     .query(f'peak_params == "{periodic_type}"')
                                     .query('tinnitus == True'))
                else:
                    cur_df = (df_periodic.query('ch_name != @physio')
                                     .query(f'peak_params == "cf"')
                                     .query('tinnitus == True')) #arbitrary choice -> just need the peaks
                
            
            #get knee and fixed chans
            knee_settings = joblib.load('/mnt/obob/staff/fschmidt/resting_tinnitus/data/knee_settings.dat')
            knee_chans = knee_settings['knee']
            fixed_chans = knee_settings['fixed']
            
            cur_df = pd.concat([cur_df.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                                cur_df.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])


            if feature in ['n_peaks', 'beta']:
                df_cf = (df_periodic.query('ch_name != @physio')
                        .query(f'peak_params == "cf"')
                        .query('tinnitus == True'))
                
                df_cf = pd.concat([df_cf.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                                   df_cf.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])


                #remove train and line noise from n peaks
                cur_df['n_peaks'] = cur_df['n_peaks'] - (np.isnan(df_cf['line_noise']) == False).to_numpy().astype(int)
                cur_df['n_peaks'] = cur_df['n_peaks'] - np.logical_and(df_cf['beta'] < 17, df_cf['beta'] > 16).to_numpy().astype(int)

                cur_df['beta'][np.logical_and(df_cf['beta'] < 17, df_cf['beta'] > 16).to_numpy()] = np.nan
            
            #%% drop bad fits
            cur_df = cur_df.mask(cur_df['r_squared'] < mask_level)
            
            #%%
            mdf = self._run_lin_reg(cur_df, feature, sample_kwargs)
            ch_effects = az.summary(mdf, var_names='beta|')

            #%% save
            if feature in ['exponent', 'offset', 'knee_freq', 'n_peaks']:
                ch_effects.to_csv(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/lin_reg/{feature}.csv')
                mdf.to_netcdf(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/lin_reg/{feature}.nc')
            elif feature in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                ch_effects.to_csv(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/lin_reg/{feature}_{periodic_type}.csv')
                mdf.to_netcdf(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/lin_reg/{feature}_{periodic_type}.nc')


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
            #priors
            alpha = pm.Normal('1|', mu=0, sigma=1.5, dims="ch_name")
            beta = pm.Normal('beta|', mu=0, sigma=1, dims="ch_name")

            #likelihood
            sigma = pm.Exponential('sigma',  lam=1)
            psi = pm.Uniform('psi', 0.1, 0.9)
            observed = pm.HurdleLogNormal('tinnitus_distress',
                                          psi=psi,#pm.math.invlogit(alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature])),
                                          mu=alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature]),
                                          sigma=sigma,
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
