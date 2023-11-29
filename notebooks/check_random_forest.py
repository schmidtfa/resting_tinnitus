#%%
import joblib
from os.path import join
from os import listdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import bambi as bmb
import arviz as az
import pymc as pm
import numpy as np

# %%
INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/ada_boost/'
# %%
acc_list, importance_list, settings_list = [], [], []

for f in listdir(INDIR):

    cur_data = joblib.load(join(INDIR, f))

    acc_list.append(cur_data['acc'])
    #importance_list.append(cur_data['importance_rf'])
    settings_list.append(cur_data['settings'])
# %%
df_acc = pd.concat(acc_list)

ch_ixs, ch_names = pd.factorize(df_acc['ch_name'])

coords = {
         "ch_name": ch_names,
         "obs_id": np.arange(len(ch_ixs)),
        }

#%%

with pm.Model(coords=coords) as md:

    #hyperprior
    a_mu = pm.Normal('a_mu', 50, 10)
    a_sigma = pm.Exponential('a_sigma', lam=1)
    #prior
    alpha = pm.Normal('a|', a_mu, a_sigma, dims="ch_name")

    #likelihood
    sigma = pm.Exponential('sigma', lam=1)
    nu = pm.Exponential('nu', lam=1)
    y = pm.StudentT('y',
                  mu=alpha[ch_ixs],
                  nu=nu,
                  sigma=sigma,
                  observed=df_acc['accuracy (%)'])


#%%
with md:
    idata = pm.sample_prior_predictive()

#%%
with md:
    idata = pm.sample()

#%%
summary = az.summary(idata, var_names='a|', filter_vars='like')

#%%
az.plot_trace(idata)

#%%
with md:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True,)

#%%
az.plot_ppc(idata)


#%%
good_chs = list(df_acc.groupby('ch_name').mean()[pd.concat(acc_list).groupby('ch_name').mean()['accuracy (%)'] > 55].index)

good_chs
#%%
print(df_acc.mean())
print(df_acc.std())

#%%
df_acc.groupby('ch_name').median().hist()

#%%
df_acc.groupby('ch_name').std().hist()

#%%
g = sns.catplot(df_acc, x='ch_name', y='accuracy (%)', kind='point')
g.set_xlabels('')
g.ax.set_xticks([])
#%%
stats.pearsonr(pd.concat(acc_list).groupby('ch_name').median()['accuracy (%)'], 
               pd.concat(acc_list).groupby('ch_name').std()['accuracy (%)'])

#%%
f, ax = plt.subplots()
ax.scatter(pd.concat(acc_list).groupby('ch_name').std(), pd.concat(acc_list).groupby('ch_name').mean())
ax.set_xlabel('StandardDeviation Accuracy (%)')
ax.set_ylabel('Mean Accuracy (%)')
# %%
df_importance = pd.concat(importance_list)#.groupby(['ch_name', 'feature']).mean()#.hist()
# %%
#df_importance.mean().sort_values(ascending=False)

# %%
df_importance.groupby('feature').mean().sort_values('importance_rf', ascending=False)
# %%
df_settings = pd.concat(settings_list)

# %%
plt.hist(df_settings['n_estimators'])

# %%
