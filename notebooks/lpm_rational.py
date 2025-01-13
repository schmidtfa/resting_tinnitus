#%% Imports
import pandas as pd
import bambi as bmb
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

#%% funciton to extract predicted probabilities from linear probability model
def predict_ldm(md, fit, y_key): 
    md.predict(fit)
    lin_preds = fit.posterior.mu.mean(axis=(0, 1))

    c = fit.posterior.Intercept.mean()
    rss = np.sum((lin_preds - fit.observed_data[y_key])**2)
    k = fit.observed_data[y_key].shape[0] / rss
    m = fit.observed_data[y_key].mean()
    a = np.log(m/(1-m)) + k*(c-.5) + .5*(1/m - 1/(1-m))
    
    
    my_preds_logit = k*(lin_preds-c) + a
    my_preds = 1/(1 + np.exp(-my_preds_logit))

    return my_preds

#%% Example 1
mroz = pd.read_stata("http://fmwww.bc.edu/ec-p/data/wooldridge/mroz.dta")

#%%
mroz_md = bmb.Model('inlf ~ kidslt6 + age + educ + exper + expersq', data = mroz)
mroz_fit = mroz_md.fit()
az.summary(mroz_fit)
#%%
mroz_md.predict(mroz_fit)
linear_pred = mroz_fit.posterior.mu.mean(axis=(0, 1))
#%%
ldm_pred = predict_ldm(mroz_md, mroz_fit, 'inlf')

#%%
logit_md = bmb.Model('inlf ~ kidslt6 + age + educ + exper + expersq', 
                      data = mroz, link='logit', family='bernoulli')

logit_fit = logit_md.fit()
az.summary(logit_fit)

#%%
logit_md.predict(logit_fit)
logit_pred = logit_fit.posterior.p.mean(axis=(0, 1))

#%%
f, ax = plt.subplots(figsize=(4,4))
ax.scatter(logit_pred, linear_pred)
ax.set_ylabel('Fitted values from Linear Model')
ax.set_xlabel('Logistic Predicted Probabilities')
stats.pearsonr(logit_pred, linear_pred)


#%%
f, ax = plt.subplots(figsize=(4,4))
ax.scatter(logit_pred, ldm_pred)
ax.set_ylabel('LDM Predicted Probabilities')
ax.set_xlabel('Logistic Predicted Probabilities')
stats.pearsonr(logit_pred, ldm_pred)


# %%
