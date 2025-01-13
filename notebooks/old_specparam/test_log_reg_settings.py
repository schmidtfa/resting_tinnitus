#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr 
from scipy.special import expit
import pymc as pm
from scipy.stats import zscore
import numpy as np
import arviz as az
from sklearn.datasets import make_classification

#%%
X, y = make_classification(n_features=1,
                           n_informative=1, 
                           n_redundant=0, 
                           n_classes=2, n_clusters_per_class=1)

#%%

cur_df = pd.DataFrame({'effect': np.concatenate(X),
                       'tinnitus': y})


#cur_df = pd.DataFrame({'effect': np.random.normal(size=106),
 #                      'tinnitus': np.concatenate([np.zeros(53), np.ones(53)])})

#%%
import bambi as bmb

md = bmb.Model('tinnitus ~ scale(effect)', data=cur_df, family='bernoulli', link='logit')
md.build()

#%%
mdf = md.fit(idata_kwargs= {'log_likelihood': True})

#%%
with pm.Model() as glm:

    #Priors
    alpha = pm.Normal('intercept', 0, 1.5)
    beta = pm.Normal('beta', 0, .5)

    #likelihood
    observed = pm.Bernoulli('tinnitus',
                            p=pm.math.invlogit(alpha + beta*zscore(cur_df['effect'])),
                            observed=cur_df['tinnitus'])


#%%
with glm:
    idata_prior = pm.sample_prior_predictive()

#%%
_, ax = plt.subplots(figsize=(5, 5))
x = xr.DataArray(np.linspace(-4, 4, 100), dims=["plot_dim"])
prior = idata_prior.prior

y = expit(prior["intercept"] + prior["beta"] * x)
y_mu = np.squeeze(expit(prior["intercept"] + prior["beta"] * x).mean(axis=1))


ax.plot(x, y.stack(sample=("chain", "draw")), c="k", alpha=0.25)
ax.plot(x, y_mu, c="r", alpha=1, linewidth=4)
ax.set_ylabel("probability (Tinnitus)")
ax.set_xlabel("Predictor (stdz)")

# %%
with glm:
    idata = pm.sample(idata_kwargs= {'log_likelihood': True}, )

# %%
az.summary(idata)
# %%
_, ax = plt.subplots(figsize=(5, 5))
x = xr.DataArray(np.linspace(-2, 2, 100), dims=["plot_dim"])
posterior = idata.posterior

y = expit(posterior["intercept"] + posterior["beta"] * x)
y_mu = np.squeeze(expit(posterior["intercept"] + posterior["beta"] * x).mean(axis=1).mean(axis=0))


ax.plot(x, y.stack(sample=("chain", "draw")), c="k", alpha=0.25)
ax.plot(x, y_mu, c="r", alpha=1, linewidth=4)
ax.set_ylabel("probability (Tinnitus)")
ax.set_xlabel("Predictor (stdz)")
# %%

with glm:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
# %%

az.plot_ppc(idata)
# %%
df_comp_loo = az.compare({"bambi": mdf, 
                          "pymc": idata})


az.plot_compare(df_comp_loo, insample_dev=False);
# %%
