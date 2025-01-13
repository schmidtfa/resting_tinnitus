#%%
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

import matplotlib.pyplot as plt
import seaborn.objects as so
import seaborn as sns
# %%

df = pd.read_csv('../data/tinnitus_match.csv').query('tinnitus == True')
# %%
df
# %%
(so.Plot(data=df, x='tinnitus_distress', y='measurement_age').add(so.Dots()))
# %%
(so.Plot(data=df, x='tinnitus_distress', y='dB').add(so.Dots()))


#%%
with pm.Model() as hurdle_log:

    # Priors
    alpha = pm.Normal('alpha', 0, 1)
    beta = pm.Normal('beta', 0, 1)

    hu_alpha = pm.Normal('hu_alpha', 0, 1)
    hu_beta = pm.Normal('hu_beta', 0, 1)    

    #sigma
    sigma = pm.Exponential('sigma', lam=1)
    #likelihood
    #link functions set as brms defaults ()
    y = pm.HurdleLogNormal('distress',
                           psi=pm.math.invlogit(hu_alpha + hu_beta*df['dB_z']),
                           mu=alpha + beta*df['dB_z'],
                           sigma=sigma,
                           observed=df['tinnitus_distress'])

#%% do prior predictive checks
with hurdle_log:
    idata = pm.sample_prior_predictive(samples=100)

#%% plot the prior predictive checks -> see if model explores the data properly
import xarray as xr 
from scipy.special import expit
_, ax = plt.subplots(ncols=2, figsize=(8, 4))

x = xr.DataArray(np.linspace(-3, 3, 100), dims=["plot_dim"])
prior = idata.prior

y = prior["alpha"] + prior["beta"] * x
y_psi = expit(prior["hu_alpha"] + prior["hu_beta"] * x)


ax[0].plot(x, y_psi.stack(sample=("chain", "draw")), c="k", alpha=0.4)
ax[0].set_ylabel("Mean Outcome (Tinnitus Distress)")
ax[0].set_xlabel("Predictor (stdz)")
ax[0].set_title("Hurdle - Part");

ax[1].plot(x, y.stack(sample=("chain", "draw")), c="k", alpha=0.4)
ax[1].set_ylabel("Mean Outcome (Tinnitus Distress)")
ax[1].set_xlabel("Predictor (stdz)")
ax[1].set_title("LogNormal - Part");

#%% 
with hurdle_log:
    idata = pm.sample()

#%% now check if the model explored the data correctly (ok this looks shit)
with hurdle_log:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)#


#%% use arviz to plot
g = az.plot_ppc(idata)
g.set_xlim(0, 24)

#%% lets look at the output
summary = az.summary(idata)


np.exp(summary.loc['hu_beta']['mean'])



# %%
(so.Plot(df, "tinnitus_distress").add(so.Bars(), so.Hist(), so.Stack(), color='gender')
                                 .label(y='Age (years)', x='Tinnitus Distress'))
# %%
sns.catplot(df, x='gender', y='tinnitus_distress', kind='point')
# %%
