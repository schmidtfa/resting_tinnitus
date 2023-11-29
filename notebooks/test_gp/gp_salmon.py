#%%
import pymc as pm
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
# %%
df = pd.read_csv('salmon.csv', sep='\s+')
# %%
plt.scatter(df['spawners'], df['recruits'])
# %%
with pm.Model() as lin_reg:

    #prior
    alpha = pm.Normal('alpha', 0, 50)
    beta = pm.Normal('beta', 0, 50)

    sigma = pm.HalfNormal('sigma', 50)

    eta = pm.HalfCauchy('eta', 3)
    rho = pm.HalfCauchy('rho', 3)
    kernel_function = eta * pm.gp.cov.ExpQuad(input_dim=1, ls=rho)
    GP = pm.gp.Latent(cov_func=kernel_function)
    alpha_gp = GP.Marginal("alpha_gp", X=df['spawners'])


    #likelihood
    pm.Normal('y',
             mu=alpha_gp,# + beta * df['spawners'],
             sigma=sigma,
             observed=df['recruits'])
# %%
with lin_reg:
    prior_pred = pm.sample_prior_predictive()
# %%
X = xr.DataArray(np.linspace(0, 500, 4000), dims=["plot_dim"])
prior = prior_pred.prior

y =prior['alpha'] + prior['beta'] * X
# %%
plt.plot(X, y.T.stack(sample=("chain", "draw")));
# %%
with lin_reg:
    idata = pm.sample()
# %%
az.summary(idata)
#%%
az.plot_trace(idata)

#%%
posterior = idata.posterior
y = posterior['alpha'] + posterior['beta'] * X

#%%
plt.plot(X, y.mean(axis=0).T, color='k', alpha=0.15)
plt.plot(X, y.mean(axis=0).T.mean(axis=1), color='r', linewidth=2)
plt.scatter(df['spawners'], df['recruits'])
# %%



#%%