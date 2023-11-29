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
from functools import partial

import seaborn as sns

import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist



df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
subject_list = list(df_all['subject_id'].unique())


sample_kwargs = {#'progressbar':False,
                'draws': 1000,
                'tune': 2000,
                'chains': 4,
                'target_accept': 0.99,}


low_freq=0.25
up_freq=98
mask_level=.9
feature = 'exponent'

all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam_3').glob(f'*/*__peak_threshold_2__freq_range_[[]{low_freq}, {up_freq}[]].dat'))
#all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam').glob(f'*/*__peak_threshold_2.5__freq_range_[[]0.25, 98[]].dat'))


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
physio = ['ECG', 'EOGV', 'EOGH']

if feature in ['exponent', 'offset', 'knee_freq']:
    cur_df = (df_aperiodic.query('ch_name != @physio')
                            )

elif feature in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'n_peaks']:
    if feature != 'n_peaks':
        cur_df = (df_periodic.query('ch_name != @physio')
                            .query(f'peak_params == "{periodic_type}"'))
    else:
        cur_df = (df_periodic.query('ch_name != @physio')
                                .query(f'peak_params == "cf"')) #arbitrary choice -> just need the peaks
    

#cur_df = knee_or_fixed(cur_df) #get fixed or knee model
#%%
no_knee = True

if no_knee:
    cur_df = cur_df.query('aperiodic_mode == "fixed"')

else:    
    knee_settings = joblib.load('/mnt/obob/staff/fschmidt/resting_tinnitus/data/knee_settings.dat')
    knee_chans = knee_settings['knee']
    fixed_chans = knee_settings['fixed']
    
    cur_df = pd.concat([cur_df.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                        cur_df.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])


if feature in ['n_peaks', 'beta']:
    df_cf = (df_periodic.query('ch_name != @physio')
                        .query(f'peak_params == "cf"'))
    if no_knee:
        df_cf = df_cf.query('aperiodic_mode == "fixed"')

    else:    
        df_cf = pd.concat([df_cf.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                            df_cf.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])


    #remove train and line noise from n peaks
    cur_df['n_peaks'] = cur_df['n_peaks'] - (np.isnan(df_cf['line_noise']) == False).to_numpy().astype(int)
    cur_df['n_peaks'] = cur_df['n_peaks'] - np.logical_and(df_cf['beta'] < 17, df_cf['beta'] > 16).to_numpy().astype(int)

    cur_df['beta'][np.logical_and(df_cf['beta'] < 17, df_cf['beta'] > 16).to_numpy()] = np.nan

#%% drop bad fits
cur_df_f = cur_df.mask(cur_df['r_squared'] < mask_level)

#%% merge with cortex labels and pick cur label for pooling
df_regions_info = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/regions_hcmp.csv')
df_regions_info['ch_name'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
df_regions_info['ch_name'] = df_regions_info['ch_name'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                                    'R_7Pl_ROI': 'R_7PL_ROI',})

df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']
# %%
df_cmb = cur_df_f.merge(df_regions_info, on='ch_name')

#%% compute a distance matrix for all regions

cur_df = df_cmb[[feature, 'tinnitus', 'ch_name', 'cortex_info', 'subject_id']].dropna()




ch_ixs, channel = pd.factorize(cur_df['ch_name'])
coords = {
    "ch_name": channel,
    "obs_id": np.arange(len(ch_ixs)),
}


df_dist = df_regions_info.copy()[['ch_name', 'cortex_info', 'x-cog', 'y-cog', 'z-cog']]
reindex_array_ch = [np.argmax(ch == channel) for ch in df_dist['ch_name']]


distance_matrix = squareform(pdist(df_dist.iloc[reindex_array_ch][['x-cog', 'y-cog', 'z-cog']])) / 10 # in cm is easier to sample

plt.imshow(distance_matrix)

standardize = lambda x : (x - np.nanmean(x)) / 2*np.nanstd(x)

#cur_df['tinnitus'].astype(int)


#%%

# cur_df_cut = cur_df.groupby(['cortex_info', 'subject_id', 'tinnitus']).mean().reset_index()


# ch_ixs, channel = pd.factorize(cur_df_cut['cortex_info'])
# coords = {
#     "ch_name": channel,
#     "obs_id": np.arange(len(ch_ixs)),
# }

# df_cortex_dist = df_dist.groupby('cortex_info').mean().reset_index()

# reindex_array_cortex = [np.argmax(ch == channel) for ch in df_cortex_dist['cortex_info']]


# #%% distance 2
# distance_matrix_2 = squareform(pdist(df_cortex_dist.iloc[reindex_array_cortex][['x-cog', 'y-cog', 'z-cog']])) / 10 # in cm is easier to sample

# plt.imshow(distance_matrix_2)

# d_std = distance_matrix_2 / distance_matrix_2.max()

with pm.Model(coords=coords) as glm_gp:

    # #Priors for Regression part
    alpha = pm.Cauchy('alpha', 0, 10)
    alpha_sigma = pm.HalfCauchy('alpha_s', 2.5)
    beta = pm.Cauchy('beta', 0, 2.5, dims='ch_name')
    beta_sigma = pm.HalfCauchy('beta_s', 2.5, dims='ch_name')

    # Priors for Gaussian Process (distance of parcels)
    eta = pm.HalfCauchy("eta", 2.5)
    rho_params = pm.find_constrained_prior( 
         pm.HalfCauchy,
         lower=0, upper=20, #we know tha the max distance between channels is 26cm
         init_guess={'beta': 1}
    )
    rho = pm.HalfCauchy("rho", **rho_params)#sets the lengthscale
    gp = pm.gp.HSGP(m=[300], c=1.5, cov_func=eta**2 * pm.gp.cov.Matern52(1, ls=rho))
    K = gp.prior("f_0", X=distance_matrix, dims='ch_name')

    # combine gp with coeffs
    alpha_mu = pm.Deterministic(
        "alpha|", alpha + alpha_sigma * K, dims="ch_name"
    )
    beta_mu = pm.Deterministic(
        "beta|", beta + beta_sigma * K, dims="ch_name"
    )
    #likelihood
    y = pm.Bernoulli('y',
                   p=pm.math.invlogit((alpha_mu[ch_ixs] + beta_mu[ch_ixs]*standardize(cur_df[feature]))),
                   observed=cur_df['tinnitus'],
                   dims='obs_id'
                   ) 
    
    #mdf = sample_numpyro_nuts(**sample_kwargs)
    #mdf =  pm.sample(**sample_kwargs)
    #mdf = pm.find_MAP()

pm.model_to_graphviz(glm_gp)

#%%
with glm_gp:
    prior_predictive = pm.sample_prior_predictive()


#%% lets check our covariance expectations
from gp_utils import plot_predictive_covariance
plot_predictive_covariance(prior_predictive.prior, label='prior', max_distance=10, color='C0')
plt.ylim(0, 5)
#%%
with glm_gp:
    map_b = pm.find_MAP()

#%%
map_b


#%%
with glm_gp:
    mdf3 = pm.sample(**sample_kwargs)


#%%


plot_predictive_covariance(prior_predictive.prior, label='prior', max_distance=100, color='C0')
plot_predictive_covariance(mdf3.posterior, label='posterior', max_distance=100, color='C1')
plt.ylim([0, 100]);
plt.title("Prior Covariance Functions");


# %%
summary = az.summary(mdf3, var_names=[ 'alpha|'])
summary

#%%
summary = az.summary(mdf3, var_names=['beta|'])
summary
#%%
summary = az.summary(mdf3)
summary

#%%
az.plot_trace(mdf3, backend_kwargs={"layout": "constrained"})

#%%
import bambi as bmb

md = bmb.Model('scale(exponent) ~ 1 + tinnitus + (1 + tinnitus|cortex_info)', data=cur_df_cut, dropna=True)

mdf2 = md.fit()
# %%
sns.set_style('ticks')
sns.set_context('poster')
plot_predictive_covariance(prior_predictive.prior, color='k', label='prior')
plot_predictive_covariance(mdf3.posterior, label='posterior')
plt.ylim([0, 2]);
# %%
az.plot_trace(mdf3, backend_kwargs={"layout": "constrained"})
# %%
summary[summary['mean'] > .158]
# %%
