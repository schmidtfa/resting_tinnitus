#%%
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import zscore

from ggseg_py.ggseg_py import rda2gpd
from ggseg_py.plotting_utils import plot_aseg
from pathlib import Path


# %%
data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_cc_final_final')

atlas_path = "/home/schmidtfa/git/ggseg_py/ggseg_py/atlases/aseg.rda"

gdf = rda2gpd('aseg', path2atlas=atlas_path)

gdf['region'] = gdf['region'].fillna('???')

cur_feat = 'gm_aseg_volume_norm' # Exponent_1, Exponent_2, tau, alpha_cf
df = pd.read_csv(data_f / f'{cur_feat}.csv', index_col = 0).reset_index()
df['label'] = [i[8:-1] for i in df['index']]

df['label'] = df['label'].replace({'3rd-Ventricle': 'x3rd-ventricle',
                            '4th-Ventricle': 'x4th-ventricle',
                            'Left-Thalamus': 'Left-Thalamus-Proper',
                            'Right-Thalamus': 'Right-Thalamus-Proper', 
                           })
rope_l, rope_h = 0 , 0
mask_neg = np.logical_and(df['hdi_94.5%'] < rope_l, df['hdi_5.5%'] < rope_l)
mask_pos = np.logical_and(df['hdi_94.5%'] > rope_h, df['hdi_5.5%'] > rope_h)

mask = (mask_pos + mask_neg).astype(int)
df['mean_mask'] = df['mean']
df['mean_mask'][mask == 0] = 0

gdf2plot = gdf.merge(df,  on='label', how='outer')
#gdf2plot['mean'] = gdf2plot['mean'].fillna(0)

f, ax = plot_aseg(gdf2plot, 
                  value='mean_mask', 
                  cmap='RdBu_r',
                  vmin=-.5,
                  vmax=0.5, 
                  show_cbar=True)
f.savefig('../results/aseg_regions_age.svg')


#%%
df_aseg = pd.read_csv(f'/home/schmidtfa/experiments/brain_age/data/data_cam_can/aseg_volume.csv')
df_gm_wm = pd.read_csv(f'/home/schmidtfa/experiments/brain_age/notebooks/brain_vols_wm_gm.csv')


df_gm_wm = df_gm_wm.merge(df_aseg[['subject', 'age']], on='subject')

# %%
df_wm = df_gm_wm.query('Name == "CerebralWhiteMatter"').drop_duplicates('subject')
# %%
df_wm
# %%
import numpy as np
import pymc as pm
import arviz as az

age = df_wm["age"].to_numpy()
wm  = df_wm["vol_norm"].to_numpy()

# Standardize age + outcome so slopes are standardized betas
age_mu, age_sd = age.mean(), age.std(ddof=0)
wm_mu,  wm_sd  = wm.mean(),  wm.std(ddof=0)

age_z = (age - age_mu) / age_sd
wm_z  = (wm  - wm_mu)  / wm_sd

# Prior center for cp around 45y, but cp is estimated from data
cp0_z = (45.0 - age_mu) / age_sd

n_draws_prior = 200
#%%
with pm.Model() as m:
    # Changepoint on standardized age scale (easier for sampling)
    cp_z = pm.Normal("cp_z", mu=cp0_z, sigma=10)  
    cp_age = pm.Deterministic("cp_age", cp_z * age_sd + age_mu) #age cp in original units

    beta1 = pm.Normal("beta1", 0, 1)
    beta2 = pm.Normal("beta2", 0, 1)

    alpha = pm.Normal("alpha", 0, 1)  # mean of wm_z at the changepoint (since model uses (age_z - cp_z))

    # Smooth transition width (in SD units of age)
    k = pm.HalfNormal("k", sigma=1)

    x = age_z
    s = pm.math.sigmoid((x - cp_z) / k)

    # Continuous piecewise linear:
    # for x << cp: slope ~ beta1
    # for x >> cp: slope ~ beta2
    mu = pm.Deterministic("mu", alpha + (x - cp_z) * (beta1 + (beta2 - beta1) * s))

    sigma = pm.Exponential('sigma', lam=1)
    nu = pm.Gamma('nu', 2, 0.1)
    y_hat = pm.StudentT("y_hat", mu=mu, nu=nu, sigma=sigma, observed=wm_z)

    priors = pm.sample_prior_predictive(draws=n_draws_prior)

#%%
import xarray as xr

p = priors.prior.stack(sample=("chain", "draw"))


age_z_sim = xr.DataArray(
    np.linspace(-2, 2, n_draws_prior),
    dims="age_sim"
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

s = sigmoid((age_z_sim  - p['cp_z']) / p['k'])
mu_prior = (p['alpha'] 
            + (age_z_sim- p['cp_z']) 
            * (p['beta2'] + (p['beta1'] - p['beta1']) * s))


#mu_prior = mu_prior.transpose("sample", "age_sim")


fig, ax = plt.subplots(figsize=(5, 4))
# prior curves
#for i in range(mu_prior.sizes["sample"]):
ax.plot(
        age_z_sim,
        mu_prior.T,
        alpha=0.1,
        linewidth=1
    )

ax.plot(
        age_z_sim,
        np.mean(mu_prior.T, axis=1),
        alpha=1,
        linewidth=2
    )

ax.set_xlabel("Age")
ax.set_ylabel("wm_z")
ax.set_title("Prior predictive check: prior mean curves")
ax.legend()
plt.show()

#%%

with m:
    idata = pm.sample(target_accept=0.95)
    ppc = pm.sample_posterior_predictive(idata)


# %%
az.summary(idata, var_names=["beta1", "beta2", "cp_age", "k", "sigma"], hdi_prob=.89)

#%% lets get the actual untis
import numpy as np
import matplotlib.pyplot as plt

sns.set_theme(context='poster',
              style='ticks',
              )

# --- inputs you already have ---
# age: 1D array in years
# wm:  1D array in original units
# age_mu, age_sd, wm_mu, wm_sd: scalers used for standardization
# idata: InferenceData returned by pm.sample()

# --- helper: posterior mean function on standardized scale ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Flatten chains/draws to a single "samples" dimension
def flat(idata, name):
    return idata.posterior[name].values.reshape(-1)

alpha = flat(idata,"alpha")
beta1 = flat(idata,"beta1")
beta2 = flat(idata,"beta2")
cp_z  = flat(idata,"cp_z")
k     = flat(idata,"k")

# Grid for plotting (in YEARS)
age_grid = np.linspace(age.min(), age.max(), 200)
x_z = (age_grid - age_mu) / age_sd  # standardized age grid

# Compute mu_z for each posterior draw and each x in the grid
# shape: (n_samples, n_grid)
s = sigmoid((x_z[None, :] - cp_z[:, None]) / k[:, None])
mu_z = alpha[:, None] + (x_z[None, :] - cp_z[:, None]) * (beta1[:, None] + (beta2[:, None] - beta1[:, None]) * s)

# Back-transform to original WM units
mu = mu_z * wm_sd + wm_mu

# Summaries for line + band
mu_mean = mu.mean(axis=0)
mu_lo, mu_hi = np.quantile(mu, [0.055, 0.945], axis=0)  
# or use 0.025/0.975 for 95%

# Changepoint in years (posterior)
cp_age = flat(idata,"cp_age")
cp_mean = cp_age.mean()
cp_lo, cp_hi = np.quantile(cp_age, [0.055, 0.945])

# --- plot ---
f, ax = plt.subplots(figsize=(7, 5))
ax.scatter(age, wm, s=18, alpha=0.4, color='#555555')

ax.plot(age_grid, mu_mean)
ax.fill_between(age_grid, mu_lo, mu_hi, alpha=0.5)

# Changepoint vertical line + interval
ax.axvline(cp_mean, linestyle="--", color='r')
ax.axvspan(cp_lo, cp_hi, alpha=0.15, color='r')

ax.set_ylim(0.25, .4)
ax.set_xlabel("Age (years)")
ax.set_ylabel("White matter volume \n (normalized by TIV)")
f.tight_layout()
f.savefig('global_white_matter_change.svg')
sns.despine()

# %%
df_gm = df_gm_wm.query('Name == "Cortex"').drop_duplicates('subject')
# %%
df_gm
# %%
with pm.Model() as gm_md:

    alpha = pm.Normal("alpha", 0, 1)
    beta = pm.Normal("beta", 0, 1)


    sigma = pm.Exponential('sigma', lam=1)
    nu = pm.Gamma('nu', 2, 0.1)
    y_hat = pm.StudentT('y_hat',
                      mu=alpha + beta * zscore(df_gm['age']),
                      nu=nu,
                      sigma=sigma,
                      observed=zscore(df_gm['vol_norm'])
                      )
    
    idata_gm = pm.sample()
# %%

az.summary(idata_gm)
# %%
alpha = flat(idata_gm,"alpha")
beta = flat(idata_gm, "beta")
# %%

age_grid = np.linspace(df_gm['age'].min(), df_gm['age'].max(), 200)

x_z = (age_grid - df_gm['age'].mean()) / df_gm['age'].std()  # standardized age grid
mu_z = alpha[:, None] + x_z[None, :] * beta[:,None]
# %%

mu = mu_z * df_gm['vol_norm'].std() + df_gm['vol_norm'].mean()

mu_mean = mu.mean(axis=0)
mu_lo, mu_hi = np.quantile(mu, [0.055, 0.945], axis=0)  
# %%
f, ax = plt.subplots(figsize=(7,5))
ax.plot(age_grid, mu_mean)
ax.fill_between(age_grid, mu_lo, mu_hi, alpha=0.5)
ax.scatter(df_gm['age'], df_gm['vol_norm'], s=18, alpha=0.4, color='#555555')

ax.set_xlabel("Age (years)")
ax.set_ylabel("Grey matter volume \n (normalized by TIV)")
f.tight_layout()
f.savefig('global_grey_matter_change.svg')
sns.despine()
# %%
