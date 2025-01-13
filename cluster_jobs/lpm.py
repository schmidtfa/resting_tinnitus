#%%
from pathlib import Path
import joblib
import pandas as pd
import pymc as pm
import numpy as np
from scipy.stats import zscore

from plus_slurm import Job
import arviz as az
from scipy.spatial.distance import squareform, pdist

# %%
class LPM(Job):

    def run(self,
            feature,
            model_type
            ):

            sample_kwargs = {#'progressbar':False,
                            'draws': 2000,
                            'tune': 3000,
                            'chains': 4,
                            'target_accept': 0.99,
                            'nuts_sampler':'numpyro'}

            #feature = 'Exponent_2'
            df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
            df_pe = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params.csv')

            cur_df_f = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])
            cur_df_f['old'] = cur_df_f['age'] > 50
            #%% merge with cortex labels and pick cur label for pooling
            df_regions_info = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/regions_hcmp.csv')
            df_regions_info['ch_name'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
            df_regions_info['ch_name'] = df_regions_info['ch_name'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                                             'R_7Pl_ROI': 'R_7PL_ROI',})

            df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']

            df_cmb = df_regions_info.merge(cur_df_f, on='ch_name') #This order is fucking important!!! TODO: check every merge
            #df_cmb = cur_df_f.merge(df_regions_info, on='ch_name') 
            #TODO: Think of a better informed way of dropping duplicate oscillations e.g. pick highest per sub and channel amplitude
            #df_cmb.drop_duplicates(subset=['subject_id', 'ch_name'], inplace=True)

            print('Running model')

            if model_type == 'hi':
                mdf = self._run_hierarchical_lpm(df_cmb, feature, df_regions_info, sample_kwargs)
                ch_effects = az.summary(mdf, 
                                        var_names='beta_ch', 
                                        hdi_prob=.89)
                
            elif model_type == 'up':
                mdf = self._run_unpooled_lpm(df_cmb, feature, df_regions_info, sample_kwargs)
                ch_effects = az.summary(mdf, 
                                        var_names='beta_ch', 
                                        hdi_prob=.89)
                
            elif model_type == 'gp':
                mdf = self._run_gp_lpm(df_cmb, feature, df_regions_info, sample_kwargs)
                ch_effects = az.summary(mdf, 
                                        var_names='beta_ch', 
                                        hdi_prob=.89)
            #% save
            ch_effects.to_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/lpm_duration_6/{feature}_{model_type}.csv')
            mdf.to_netcdf(f'/home/schmidtfa/experiments/resting_tinnitus/data/lpm_duration_6/{feature}_{model_type}.nc')
            #ch_effects.to_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/lpm_age/{feature}_{model_type}.csv')
            #mdf.to_netcdf(f'/home/schmidtfa/experiments/resting_tinnitus/data/lpm_age/{feature}_{model_type}.nc')


    # %% define regression model
    def _run_hierarchical_lpm(self, df, feature, df_regions_info, sample_kwargs):

        cur_df = df[['cortex_info', 'ch_name', feature, 'tinnitus', 'old', 'subject_id']].dropna()

        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

        ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
        coords['cortices'] = cortices
    
        with pm.Model(coords=coords) as glm:

            #l1 cortices
            alpha_m1 = pm.StudentT('alpha_m1', nu=5, mu=0, sigma=1.5, dims='cortices')
            beta_m1 = pm.StudentT('beta_m1', nu=5, mu=0, sigma=1.5, dims='cortices')

            alpha_s1 = pm.Gamma('alpha_s1', 2, .5, dims='cortices')
            beta_s1 = pm.Gamma('beta_s1', 2, .5, dims='cortices')

            #l2 parcels nested in cortices
            alpha_m2 = pm.StudentT('alpha_m2', nu=5, mu=0, sigma=1.5, dims='ch_name')
            beta_m2 = pm.StudentT('beta_m2', nu=5, mu=0, sigma=1.5, dims='ch_name')

            alpha_ch = pm.Deterministic('alpha_ch', alpha_m1[ch2cort_ixs] + alpha_m2 * alpha_s1[ch2cort_ixs], dims='ch_name')
            beta_ch = pm.Deterministic('beta_ch', beta_m1[ch2cort_ixs] + beta_m2 * beta_s1[ch2cort_ixs], dims='ch_name')

            # #likelihood
            sigma = pm.Exponential('sigma', lam=1)
            y = pm.Normal('y',
                             mu=alpha_ch[ch_ixs] + beta_ch[ch_ixs]*standardize(cur_df[feature]),
                             observed=cur_df['tinnitus'],
                             sigma=sigma,
                             dims='obs_id'
                             ) 

            mdf =  pm.sample(**sample_kwargs)


        return mdf#, glm
    


    def _run_unpooled_lpm(self, df, feature, df_regions_info, sample_kwargs):

        cur_df = df[['cortex_info', 'ch_name', feature, 'tinnitus', 'old', 'subject_id']].dropna()
        #cur_df.drop_duplicates(subset=['subject_id', 'ch_name'], inplace=True)
        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

        ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
        coords['cortices'] = cortices
    
        with pm.Model(coords=coords) as glm:

            #l2 parcels
            alpha_ch = pm.StudentT('alpha_ch', nu=5, mu=0, sigma=.5, dims='ch_name')
            beta_ch = pm.StudentT('beta_ch', nu=5, mu=0, sigma=.5, dims='ch_name')

            # #likelihood
            sigma = pm.HalfStudentT('sigma', nu=5, sigma=0.5)
            y = pm.Normal('y',
                            mu=alpha_ch[ch_ixs] + beta_ch[ch_ixs]*standardize(cur_df[feature]),
                            observed=cur_df['tinnitus'],
                            sigma=sigma,
                            dims='obs_id'
                            ) 

            mdf =  pm.sample(**sample_kwargs)


        return mdf#, glm
    


    def _run_gp_lpm(self, df, feature, df_regions_info, sample_kwargs):

        cur_df = df[['cortex_info', 'ch_name', feature, 'tinnitus', 'old', 'subject_id']].dropna()
        #cur_df.drop_duplicates(subset=['subject_id', 'ch_name'], inplace=True)

        ch_ixs, channel = pd.factorize(cur_df['ch_name'])
        coords = {
            "ch_name": channel,
            "obs_id": np.arange(len(ch_ixs)),
        }

        df_dist = df_regions_info.copy()[['ch_name', 'x-cog', 'y-cog', 'z-cog']]
        reindex_array_ch = [np.argmax(ch == channel) for ch in df_dist['ch_name']]
        distance_matrix = squareform(pdist(df_dist.iloc[reindex_array_ch][['x-cog', 'y-cog', 'z-cog']])) / 10 # in cm is easier to sample

        standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

        ch2cort_ixs, cortices = pd.factorize(df_regions_info['cortex_info'])
        coords['cortices'] = cortices
        #min_dist = distance_matrix[distance_matrix > 0].min()

        m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=[distance_matrix.min(), 
                 distance_matrix.max()], #imput what we know based on our distance matrix
        lengthscale_range=[.1, 5], #95% should fall in that range
        cov_func="Matern52"
        )

        print("Recommended smallest number of basis vectors (m):", m)
        print("Recommended smallest scaling factor (c):", np.round(c, 1))

        with pm.Model(coords=coords) as glm:

            # Priors for Gaussian Process (distance of parcels)
            eta_guess = pm.find_constrained_prior(pm.InverseGamma,
                                                lower=0.1,
                                                upper=2,
                                                init_guess={'alpha': 5, 'beta': 10},
                                                mass=0.89)

            eta = pm.InverseGamma("eta", alpha=eta_guess['alpha'], beta=eta_guess['beta'])

            ell_guess = pm.find_constrained_prior(pm.InverseGamma,
                                                lower=0.1,
                                                upper=10,
                                                init_guess={'mu': 0.1, 'sigma': 1},
                                                mass=0.89)

            ell = pm.InverseGamma("rho", mu=ell_guess['mu'], sigma=ell_guess['sigma'])

            gp = pm.gp.HSGP(m=[m], #number of basis functions
                            c=c, #
                            cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=ell)) # model distance via ornstein-uhlenbeck
            K = gp.prior("f", X=distance_matrix, dims=('ch_name', 'ch_name')) #shape should reflect the shape of the channel dimension

            #get channel intercepts and slopes pooled to the mean relative to the spatial distance 
            alpha_mu = pm.StudentT('alpha_mu', nu=10, mu=0, sigma=.5)
            alpha_sd= pm.HalfStudentT('alpha_sd', nu=10, sigma=.5)
            alpha_ch = pm.Deterministic(
                        "alpha_ch", alpha_mu + alpha_sd * K, dims="ch_name"
                    )

            beta_mu = pm.StudentT('beta_mu', nu=10, mu=0, sigma=.5)
            beta_sd= pm.HalfStudentT('beta_sd', nu=10, sigma=.5)
            beta_ch = pm.Deterministic(
                        "beta_ch", beta_mu + beta_sd * K, dims="ch_name"
                    )

            # #likelihood
            #sigma = pm.Exponential('sigma', lam=1)
            sigma=pm.HalfStudentT('sigma', nu=5, sigma=.5)
            y = pm.Normal('y',
                            mu=alpha_ch[ch_ixs] + beta_ch[ch_ixs]*standardize(cur_df[feature]),
                            observed=cur_df['tinnitus'],
                            sigma=sigma,
                            dims='obs_id'
                            ) 

            mdf =  pm.sample(**sample_kwargs)


        return mdf#, glm

if __name__ == '__main__':

    #%
    feature = 'exponent'


    job = LPM(feature=feature)
    job.run_private()



