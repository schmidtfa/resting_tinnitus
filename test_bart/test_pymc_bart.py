#%%
import pymc as pm
import pymc_bart as pmb
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import arviz as az
import pandas as pd
from scipy.stats import zscore

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer.data.features 
#¥
#X2 = X.fillna(X.mean()) #impute nans with the columns median
y = breast_cancer.data.targets 
  
y = (y == 'M').astype(int)['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
#np.isnan(X).sum()

#%%
sample_kwargs = {
                'draws': 2000,
                'tune': 2000,
                'chains': 4,}

#%% model definition

with pm.Model() as model:
    # data containers
    model.add_coord('id', X_train.index, mutable=True)
    model.add_coord('feature', X_train.columns, mutable=True)

    X_s = pm.MutableData('X_s', X_train, dims=('id', 'feature'))
    y_s = pm.MutableData("y_s", y_train)
    
    # model definiton
    mu = pmb.BART("mu", X=X_s, Y=y_train, m=100, dims='id')
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    
    # likelihood
    y = pm.Bernoulli("y_pred", p=p, observed=y_s, dims='id')

#%% training
with model:
    # actual training via MCMC
    idata = pm.sample(**sample_kwargs)

#%% testing
with model:
    pm.set_data({"X_s": X_test, 
                 "y_s": y_test}, coords={'id': X_test.index})
    idata.extend(pm.sample_posterior_predictive(idata))
# %%
pmb.plot_convergence(idata, var_name="mu");
# %%
az.summary(idata)

# %%
pmb.plot_variable_importance(idata, mu, X_train, samples=100);
# %%
y_pred = (idata.posterior_predictive.y_pred.mean(axis=0).mean(axis=0) > 0.5).to_numpy().astype(int)
# %%
(y_test == y_pred).mean()
# %%
from sklearn.ensemble import RandomForestClassifier
# %%

clf = RandomForestClassifier(n_estimators=100) #make comparable to above


clf.fit(X_train, y_train)

#%%
clf.score(X_test, y_test)
# %%

# %%
