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


n_trees = 200
INDIR = f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/bart/n_trees_{n_trees}'


df_cmb = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_all_spec_features.csv')
chs = list(df_cmb['ch_name'].unique())
n_folds = np.arange(5)

var_name = 'p'

#%%
importance_dict, diagnostics_dict, acc_dict = {}, {}, {}

diag_list, acc_list, loo_list = [], [], []

for cur_ch in chs:


    ess_l, rhat_l, loo_l = [], [], [] 
    y_pred_l, y_test_l = [], []
    inf_list = []
    for cur_fold in n_folds:
        #cur_fold = n_folds[0]
        cur_file = az.from_netcdf(join(INDIR, f'{cur_ch}_fold_{cur_fold}.nc'))

        #%data for diagnostics plot
        ess_l.append(np.atleast_2d(az.ess(cur_file, method="bulk", var_names=var_name)[var_name].values))
        rhat_l.append(np.atleast_2d(az.rhat(cur_file, var_names=var_name)[var_name].values))

        #% loo scores 2 compare
        loo_l.append(az.loo(cur_file)[0])

        #% variable importance (concatenated HDIs)

        #% decoding accuracy 
        y_pred_l.append(cur_file.posterior_predictive.y_pred.mean(['draw', 'chain']) > 0.5)
        y_test_l.append(cur_file.observed_data.y_test_set.to_numpy())


    #% add loos
    loo_list.append(pd.DataFrame({'ch_name': cur_ch,
                                  'n_trees': n_trees,
                                  'elpd': np.array(loo_l)}))

    #% add diagnostics
    rhat = np.concatenate(rhat_l, axis=1).flatten()
    ess = np.concatenate(ess_l, axis=1).flatten()

    diag_list.append(pd.DataFrame({'rhat': rhat,
                                   'ess': ess,
                                  'ch_name': cur_ch}))

    #% add decoding df
    acc_list.append(pd.DataFrame({'y_pred': np.concatenate(y_pred_l),
                                  'y_test': np.concatenate(y_test_l),
                                  'ch_name': cur_ch}))


#%%

df_diag = pd.concat(diag_list)
diag_rhat_max = df_diag.groupby('ch_name').max().reset_index()
diag_ess_min = df_diag.groupby('ch_name').min().reset_index()
diag_ave = df_diag.groupby('ch_name').mean().reset_index()

good_rhats = set(diag_rhat_max[diag_rhat_max['rhat'] < 1.05]['ch_name'].to_list())
good_ess = set(diag_ess_min[diag_ess_min['ess'] > 400]['ch_name'].to_list())


#%%
ch_list = list(good_rhats.intersection(good_ess))

#%% check acc for ok channels
df_acc = pd.concat(acc_list)

df_acc['acc'] = df_acc['y_pred'] == df_acc['y_test']

mean_acc = df_acc.groupby('ch_name').mean()['acc']
std_acc = df_acc.groupby('ch_name').std()['acc']

#%%
#plt.scatter(mean_acc, diag_ave['ess'])

mean_acc.reset_index().query('ch_name == @ch_list').hist()

#%%
(df_acc['y_pred'] == df_acc['y_test']).mean()
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
    g = plt.hist(ess, bins=50)
    sns.despine()
    g.set_xlabel('Effective Sample Size')
    g.set_ylabel('Count')
    return g


#plot_ess(ess)

#%%
df_diag = pd.concat(diag_list)

plot_rhat(df_diag['rhat'], df_diag['ess'])

#%%
plot_ess(df_diag['ess'])
#%% data for classification accuracy

(df_diag.groupby('ch_name').mean()['rhat'] > 1.01).sum()

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
