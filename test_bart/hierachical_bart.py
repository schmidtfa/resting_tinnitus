#%%
import pymc as pm
import pymc_bart as pmb
import arviz as az
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import zscore

# %% get the sleepstudy data set
sleepstudy = pd.read_csv('https://www.picostat.com/system/files/datasets/dataset-85663.csv')

y = zscore(sleepstudy['Reaction'])
X = sleepstudy[['Days', 'Subject']]

#%%
subject_idxs, subject = pd.factorize(sleepstudy['Subject'])
coords = {
    "subject": subject,
    "obs_id": np.arange(len(subject_idxs)),
}

#%% model definition
with pm.Model(coords=coords) as multi_bart_model:

    subject_idx = pm.MutableData("subject_idx", subject_idxs, dims="obs_id")
    multi_bart_model.add_coord('feature', X.drop(columns='Subject').columns, mutable=True)
    X_s = pm.MutableData('X', zscore(sleepstudy['Days'].to_numpy())[:,None], dims=("obs_id", "feature"))
    
    # add multilevel priors
    #Hyperpriors
    mu_a = pm.Normal('intercept', 0, 1)
    sigma_a = pm.Exponential('sigma_intercept', lam=1)
    alpha = pm.Normal('1|', mu_a, sigma_a, dims="subject")
    

    mu_b = pm.Normal('beta', 0, 1,)
    sigma_b = pm.Exponential('sigma_beta', lam=1)
    beta = pm.Normal('beta|', mu_b, sigma_b, dims="subject")
    
    # model definiton
    mu = pmb.BART("mu", X=X_s, Y=y, m=10, dims='obs_id')
    mu_c = pm.Deterministic("mu_b", alpha[subject_idx] + beta[subject_idx] * mu)
    
    # likelihood
    sigma = pm.Exponential('sigma', lam=1)
    y_pred = pm.Normal("y_pred", 
                       mu=mu_c, 
                       sigma=sigma, 
                       observed=y, 
                       dims='obs_id')
#%% 

with pm.Model(coords=coords) as multi_linear_model:

    subject_idx = pm.MutableData("subject_idx", subject_idxs, dims="obs_id")
    #multi_linear_model.add_coord('feature', X.drop(columns='Subject').columns, mutable=True)
    #X_s = pm.MutableData('X', zscore(sleepstudy['Days'].to_numpy())[:,None], dims=("obs_id", "feature"))
    
    # add multilevel priors
    #Hyperpriors
    mu_a = pm.Normal('intercept', 0, 1)
    sigma_a = pm.Exponential('sigma_intercept', lam=1)
    alpha = pm.Normal('1|', mu_a, sigma_a, dims="subject")
    

    mu_b = pm.Normal('beta', 0, 1,)
    sigma_b = pm.Exponential('sigma_beta', lam=1)
    beta = pm.Normal('beta|', mu_b, sigma_b, dims="subject")
    
    # model definiton
    #mu = pmb.BART("mu", X=X_s, Y=y, m=10, dims='obs_id')
    mu_linr = pm.Deterministic("mu_linr", alpha[subject_idx] + beta[subject_idx] * zscore(X['Days']), dims='obs_id')
    
    # likelihood
    sigma = pm.Exponential('sigma', lam=1)
    y_pred = pm.Normal("y_pred", 
                       mu=mu_linr, 
                       sigma=sigma, 
                       observed=y, 
                       dims='obs_id')

#%% training
sample_kwargs = {
                'draws': 2000,
                'tune': 2000,
                'chains': 4,
                'target_accept': 0.95,
                'idata_kwargs' : {'log_likelihood': True}, 
                }


with multi_bart_model:
    # actual training via MCMC
    idata_bart = pm.sample(**sample_kwargs)

with multi_linear_model:
    # actual training via MCMC
    idata_lin = pm.sample(**sample_kwargs)


#%%
models = {

    "hierarchical bart": idata_bart,
    "hierarchical linear": idata_lin,
}
df_compare = az.compare(models)

#%%
az.plot_compare(df_compare, insample_dev=False);
# %%
pmb.plot_convergence(idata_bart, var_name="mu_b");

#%%
az.summary(idata)

#%%
az.plot_trace(idata)

# %%
with multi_model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

#%%


az.plot_ppc(idata)

# %%
