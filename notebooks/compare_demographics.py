#%%
import pandas as pd
import pymc as pm
import arviz as az
from scipy.stats import zscore




df = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
# %%

def run_model(y):
    with pm.Model() as md:

        alpha = pm.Normal('alpha', 0, 1)
        beta = pm.Normal('beta', 0, 1)

        sigma = pm.HalfNormal('sigma', 1)

        y_hat = pm.Normal('y_hat',
                        mu= alpha + beta * df['tinnitus'].astype(int),
                        sigma=sigma,
                        observed=y)

    with md:
        idata = pm.sample()

    return az.summary(idata)
# %%
az_age = run_model(df['age_z'])

#%%
az_db = run_model(df['dB_z'])

#%%
with pm.Model() as md:

    alpha = pm.Normal('alpha', 0, 1)
    beta = pm.Normal('beta', 0, 1)

    y_hat = pm.Bernoulli('y_hat',
                    p= pm.invlogit(alpha + beta * df['tinnitus'].astype(int)),
                    observed=df['gender'] == 'male')

with md:
    idata = pm.sample()

az_sex = az.summary(idata)

# %%
import numpy as np
np.exp(az_sex['mean'])

# %%
az_age

#%%
az_db
# %%
