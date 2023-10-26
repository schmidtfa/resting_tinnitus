#%%
import arviz as az
from os.path import join
import pandas as pd
import pymc_bart as pmb
import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')
# %%

INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/bart/n_trees_50_old'


df_cmb = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_all_spec_features.csv')
chs = list(df_cmb['ch_name'].unique())
n_folds = np.arange(5)

var_name = 'p'

cur_ch = chs[0]






#%%
diagnostics_dict = {}

ess_l, rhat_l = [], [] 
inf_list = []
for cur_fold in n_folds:
    #cur_fold = n_folds[0]

    cur_file = az.from_netcdf(join(INDIR, f'{cur_ch}_fold_{cur_fold}.nc'))

    inf_list.append(cur_file)







    #%data for diagnostics plot
    ess_l.append(np.atleast_2d(az.ess(cur_file, method="bulk", var_names=var_name)[var_name].values))
    rhat_l.append(np.atleast_2d(az.rhat(cur_file, var_names=var_name)[var_name].values))

#%%
az.concat(inf_list, dim='draw')


#%%
plt.hist(np.concatenate(ess_l, axis=1).flatten());

#%%
rhat = np.concatenate(rhat_l, axis=1).flatten()
ess = np.concatenate(ess_l, axis=1).flatten()


#%% R-hat plot function
def plot_rhat(rhat, ess):
    g = az.plot_ecdf(rhat)
    # Assume Rhats are N(1, 0.005) iid. Then compute the 0.99 quantile
    # scaled by the sample size and use it as a threshold.
    g.axvline(norm(1, 0.005).ppf(0.99 ** (1 / ess.size)), color="0.7", ls="--")
    sns.despine()
    g.set_xlabel('R-hat')
    g.set_ylabel('Cumulative Probability')
    return g

def plot_ess(ess):
    g = plt.hist(ess)
    sns.despine()
    g.set_xlabel('ESS (Bulk)')
    g.set_ylabel('Count')
    return g


#plot_ess(ess)

#%%
plot_rhat(rhat, ess)
#%% data for classification accuracy



#%% get variable importance
ev_mean = np.zeros(len(var_imp))
ev_hdi = np.zeros((len(var_imp), 2))
for idx, subset in enumerate(subsets):
    predicted_subset = _sample_posterior(
        all_trees=all_trees,
        X=X,
        rng=rng,
        size=samples,
        excluded=subset,
        shape=shape,
    )
    pearson = np.zeros(samples)
    for j in range(samples):
        pearson[j] = (
            pearsonr(predicted_all[j].flatten(), predicted_subset[j].flatten())[0]
        ) ** 2
    ev_mean[idx] = np.mean(pearson)
    ev_hdi[idx] = az.hdi(pearson)




# %%
pmb.plot_convergence(cur_file, var_name='p')
# %%
az.summary(cur_f, var_names='p').sort_values('r_hat')
# %%
y_pred = (cur_f.posterior_predictive.y_pred.mean(['chain', 'draw']) > 0.5).to_numpy() 
y_test = cur_f.observed_data.y_test_set.to_numpy()

(y_pred == y_test).mean()
# %%
cur_f.posterior_predictive.y_pred.mean(axis=0).mean(axis=0)
# %%
az.plot_trace(cur_f, var_names='p')
# %%
