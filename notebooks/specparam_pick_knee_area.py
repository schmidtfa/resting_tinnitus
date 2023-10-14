#%%
from pathlib import Path
import joblib
import pandas as pd
import pymc as pm
import arviz as az
import bambi as bmb
import mne
import numpy as np
from scipy.stats import zscore
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')

sample_kwargs = {'draws': 1000,
               'tune': 1000,
               'chains': 2,
               'target_accept': 0.95,}


# %%
INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam'
all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam').glob(f'*/*__peak_threshold_2.5__freq_range_[[]1, 98[]].dat'))


# %%
periodic, aperiodic = [], []

for f in all_files:
    
    cur_data = joblib.load(f)

    periodic.append(cur_data['periodic'])
    aperiodic.append(cur_data['aperiodic'])

    freq = cur_data['freq']
    label_info = cur_data['label_info']
# %% 

df_periodic = pd.concat(periodic)
df_aperiodic = pd.concat(aperiodic)
#%% Test for physiological differences in aperiodic activity (tinnitus vs. control)
physio = ['ECG', 'EOGV', 'EOGH']
df_ap_physio = df_aperiodic.query('ch_name == @physio')


import sys
sys.path.append('/mnt/obob/staff/fschmidt/resting_tinnitus/utils')

from specparam_utils import knee_or_fixed
df_ap_physio = knee_or_fixed(df_ap_physio) #get fixed or knee model

df_ap_physio = df_ap_physio.mask(df_ap_physio['r_squared'] < 0.9)

cat_plot_kwargs = {'x': 'tinnitus',
                   'col': 'ch_name',
                   'hue': 'tinnitus',
                   'kind': 'point',
                   'margin_titles': True,
                   'sharey': False}


#%% offset
sns.catplot(data=df_ap_physio, y='offset', **cat_plot_kwargs)

#%% exponent
sns.catplot(data=df_ap_physio, y='exponent', **cat_plot_kwargs)



#%%
ch_name = 'EOGV'

md = bmb.Model(data=df_ap_physio.query(f'ch_name == "{ch_name}"'),
         formula='tinnitus ~ 1 + scale(offset)',
         dropna=True,
         family='bernoulli',
         link='logit'
         )

#mdf = md.fit()

#%%
#az.summary(mdf)




#%%











# %%
# statistical model:  tinnitus ~ 1 + feature (e.g. exponent) + (1|ch_name)
import pytensor.tensor as pt

df_ap_brain = df_aperiodic.query('ch_name != @physio')
df_ap_brain = knee_or_fixed(df_ap_brain)


cur_df = df_ap_brain[['knee_modeled', 'tinnitus', 'ch_name']].dropna()
#%%
ch_ixs, channel = pd.factorize(cur_df['ch_name'])
coords = {
    "ch_name": channel,
    "obs_id": np.arange(len(ch_ixs)),
}

with pm.Model(coords=coords) as glm:

    mu_a = pm.Normal('intercept', 0, 1.5)
    z_a = pm.Normal('z_a', 0, 1.5, dims="ch_name")
    sigma_a = pm.Exponential('sigma_intercept', lam=1)


    mu_b = pm.Normal('beta', 0, 1.)
    z_b = pm.Normal('z_b', 0, 1., dims="ch_name")
    sigma_b = pm.Exponential('sigma_beta', lam=1)

    # #model correlation of predictors -> this will be needed to incorporate ecg and eog
    # chol, corr, stds = pm.LKJCholeskyCov(
    # "chol", n=2, eta=4.0, sd_dist=pm.Exponential.dist(1.0, shape=2)
    # )
    # z    = pm.Normal('z',0,1,shape=(len(channel),2)) #vals_raw in pymc docs
    # v    = pm.Deterministic('v', pt.dot(chol,z.T).T) #vals in pymc docs
    

    alpha = pm.Deterministic('1|', (mu_a + z_a * sigma_a), dims="ch_name") # + v[:,0]
    beta = pm.Deterministic('beta|', (mu_b + z_b * sigma_b), dims="ch_name") # + v[:,1]

    #likelihood
    observed = pm.Bernoulli('tinnitus',
                            p=pm.math.invlogit(alpha[ch_ixs]),# + beta[ch_ixs]*zscore(cur_df[feature])),
                            observed=cur_df['knee_modeled'],
                            dims="obs_id")

    #mdf = pm.sample(**sample_kwargs)



#%%
with glm:
    mdf = pm.sample(**sample_kwargs)

#%%
#%%
sum = az.summary(mdf, var_names=['1|'])

#%%
(np.exp(sum['hdi_3%']) > 1).sum()

#%%
(np.exp(sum['hdi_97%']) < 1).sum()


#%%
ave_knee = cur_df.groupby('ch_name').mean()['knee_modeled']

#%%
knee_settings = {'knee': list(ave_knee.index[ave_knee > 0.5]),
                 'fixed': list(ave_knee.index[ave_knee <= 0.5]),}


joblib.dump(knee_settings, '../data/knee_settings.dat')
#%%
knee_chans = knee_settings['knee']
fixed_chans = knee_settings['fixed']

pd.concat([df_aperiodic.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
           df_aperiodic.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])


#%%
with glm:
    idata = pm.sample_prior_predictive(samples=100)



#%%
import xarray as xr

_, ax = plt.subplots()

x = xr.DataArray(np.linspace(-2, 2, 50), dims=["plot_dim"])
prior = idata.prior
y = prior["intercept"] + prior["beta"] * x

ax.plot(x, y.stack(sample=("chain", "draw")), c="k", alpha=0.4)

ax.set_xlabel("Predictor (stdz)")
ax.set_ylabel("Mean Outcome (stdz)")
ax.set_title("Prior predictive checks -- Weakly regularizing priors");


#%%
plt.hist(prior["beta|"].to_numpy().mean(axis=1).flatten())

plt.hist(prior["beta|"].to_numpy().std(axis=1).flatten())



#%%
sum = az.summary(mdf, var_names=['1|'])

#%%
fig,ax = plt.subplots()
ax.hist(np.exp(sum['mean']))
ax.set_xscale('log')
#%%
(np.exp(sum['hdi_3%']) > 1).sum()

#%%
(np.exp(sum['hdi_97%']) < 1).sum()


#%%
with glm:
    pm.sample_posterior_predictive(mdf, extend_inferencedata=True)

#%%

az.plot_ppc(mdf, num_pp_samples=100);



#%%
def run_log_reg(df, feature, sample_kwargs):

    cur_df = df[[feature, 'tinnitus', 'ch_name']].dropna()

    ch_ixs, channel = pd.factorize(cur_df['ch_name'])
    coords = {
        "ch_name": channel,
        "obs_id": np.arange(len(ch_ixs)),
    }

    with pm.Model(coords=coords) as glm:

        #ch_x = pm.ConstantData("channel_idx", channel_idxs, dims="obs_id")
        #predictor = pm.ConstantData(feature, zscore(cur_df[feature]), dims="obs_id")

        mu_a = pm.Normal('intercept', 0, 1.5)
        z_a = pm.Normal('z_a', 0, 1.5, dims="ch_name")
        sigma_a = pm.Exponential('sigma_intercept', lam=1)


        mu_b = pm.Normal('beta', 0, 1)
        z_b = pm.Normal('z_b', 0, 1, dims="ch_name")
        sigma_b = pm.Exponential('sigma_beta', lam=1)

        #model correlation of predictors
        chol, corr, stds = pm.LKJCholeskyCov(
        "chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2)
        )
        z    = pm.Normal('z',0,1,shape=(len(channel),2)) #vals_raw in pymc docs
        v    = pm.Deterministic('v', pt.dot(chol,z.T).T) #vals in pymc docs

        alpha = pm.Deterministic('1|', (z_a * sigma_a) + v[:,0], dims="ch_name")
        beta = pm.Deterministic('beta|', (z_b * sigma_b) + v[:,1], dims="ch_name")

        #likelihood
        observed = pm.Bernoulli('tinnitus',
                                p=pm.math.invlogit(alpha[ch_ixs] + beta[ch_ixs]*zscore(cur_df[feature])),
                                observed=cur_df['tinnitus'],
                                dims="obs_id")

        mdf = pm.sample(**sample_kwargs)

    return mdf, glm
#%%
mdf2, glm = run_log_reg(df_ap_brain, feature, sample_kwargs)


#%%
sumb = az.summary(mdf2, var_names=['beta|'])

#%%
#%%
(sum['hdi_3%'] > 0.).sum()

#%%
az.plot_trace(mdf)

#%%

md = bmb.Model(data=df_ap_brain,
         formula='tinnitus ~ 1 + scale(exponent) + (1 + scale(exponent)|ch_name)',
         dropna=True,
         family='bernoulli',
         link='logit'
         )

#%%
md2 = bmb.Model(data=df_ap_brain,
         formula='scale(exponent) ~ 1 + scale(measurement_age) + (1 + scale(measurement_age)|ch_name)',
         dropna=True,
         )


#%%
md

#%%
mdf = md.fit()

#%%



#%%
mdf_2 = md2.fit()

#%%
az.summary(mdf_2)


#%%
mdf.to_netcdf(f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/log_reg/{feature}_{ap_mode}.nc')

#%%
sns.set_context('paper')
az.plot_trace(mdf)

# %%
ch_effects = az.summary(mdf_exp, var_names=['beta|'])



#%%
freqs = 200

df_aperiodic['knee_modeled'] = (df_aperiodic['aperiodic_mode'] == 'knee').astype(int)

#%%
df_aperiodic['num_params'] = df_aperiodic['knee_modeled'] + df_aperiodic['n_peaks']

#%%
# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


df_aperiodic['bic_ap_mode'] = calculate_bic(freqs, df_aperiodic['error'], df_aperiodic['num_params'])

#%%
for subject in df_aperiodic['subject_id'].unique():

    cur_ap = df_aperiodic.query(f'subject_id == "{subject}"')

    k = cur_ap.query('aperiodic_mode == "knee"')
    f = cur_ap.query('aperiodic_mode == "fixed"')

    #Note: this works because channels are sorted the same (but its a bit dangerous)
    #check for nans as they may result in dropped channels
    knee_chs = list(f['ch_name'][f['bic_ap_mode'] > k['bic_ap_mode']]) #knee better (smaller bic wins)
    fixed_chs = list(f['ch_name'][f['bic_ap_mode'] < k['bic_ap_mode']])

    pd.concat([k.query('ch_name == @knee_chs'), f.query('ch_name == @fixed_chs')])

    
    #knee_x_fixed = cur_ap[['ch_name', 'aperiodic_mode', 'bic_ap_mode']].pivot_table(index=['ch_name',], columns=['aperiodic_mode'], values='bic_ap_mode')

#%%

#%%
df_knee_proba = (knee_x_fixed['fixed'] > knee_x_fixed['knee']).reset_index().groupby('ch_name').mean().reset_index().query('ch_name != @physio')

df_knee_proba.columns = ['ch_name', 'p(knee > fixed)']

df_knee_proba.to_csv('../data/knee_across_brain.csv')

#%%

(df_knee_proba['p(knee > fixed)'] > 0.5).sum()
# %%


sns.catplot(df_ap_physio.query('aperiodic_mode == "knee"').query('ch_name == "ECG"').reset_index(), x='tinnitus', y='knee_freq', kind='point')
# %%
df_ap_physio.query('aperiodic_mode == "knee"').query('ch_name == "ECG"').reset_index().groupby('tinnitus').corr('spearman')
# %%


cur_df = (df_ap_physio.query('aperiodic_mode == "knee"')
                     .query('ch_name == "ECG"')[[feature, 'tinnitus', 'ch_name', 'subject_id']]
                     .drop_duplicates('subject_id')
                     .dropna())

#%%
with pm.Model() as glm:

    #ch_x = pm.ConstantData("channel_idx", channel_idxs, dims="obs_id")
    #predictor = pm.ConstantData(feature, zscore(cur_df[feature]), dims="obs_id")

    #Hyperpriors
    a = pm.Normal('intercept', 0, 1.5)
    sigma_a = pm.Exponential('sigma_intercept', lam=1)
    b = pm.Normal('beta', 0, .5)
    sigma_b = pm.Exponential('sigma_beta', lam=1)

    #likelihood
    observed = pm.Bernoulli('tinnitus',
                            p=pm.math.invlogit(a + b*zscore(cur_df[feature])),
                            observed=cur_df['tinnitus'])

    mdf = pm.sample(**sample_kwargs)

# %%
sum = az.summary(mdf)
# %%
np.exp(sum.loc['beta'])
# %%
