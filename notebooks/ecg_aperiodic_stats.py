#%%
import pandas as pd
import arviz as az
import pytensor.tensor as pt
import pymc as pm

import seaborn as sns 
import matplotlib.pyplot as plt

import numpy as np

sns.set_theme(context='poster',
              style='ticks',
              palette='deep')

import scipy.stats as st
# %%
df_ecg = pd.read_csv('ecg_aperiodic_fit.csv')
# %%

plt.scatter(df_ecg['Exponent_1'], df_ecg['age'])
st.pearsonr(np.log(df_ecg['Exponent_1']), df_ecg['age'])
# %%
plt.scatter(df_ecg['Exponent_2'], df_ecg['age'])
st.pearsonr(df_ecg['Exponent_2'], df_ecg['age'])

# %%
plt.scatter(df_ecg['tau'], df_ecg['age'])
st.pearsonr(np.log(df_ecg['tau']), df_ecg['age'])

standardize = lambda x : (x - np.nanmean(x)) / (np.nanstd(x))

df_ecg['age_z'] = standardize(df_ecg['age'])
# %%
coords = {'effect' : ["age", "tinnitus", "age:tinnitus"]}

eff_dfs = []
for feature in  ['Offset', 'Exponent_1', 'Exponent_2', 'tau']:

    if feature in ['Exponent_1', 'tau']:
        df_ecg[feature] = np.log(df_ecg[feature])

    df_ecg[feature] = standardize(df_ecg[feature])

    sample_kwargs = {#'progressbar':False,
                    'draws': 4000,
                    'tune': 4000,
                    'chains': 4,
                    #'cores': 1,
                    'target_accept': 0.99,
                    'nuts_sampler':"blackjax"
                    #'nuts_sampler':'numpyro',
                    #'idata_kwargs': {"log_likelihood": True}
                    }

    with pm.Model(coords=coords) as md:

        age = pm.Data('age', df_ecg['age_z'].values)
        tinnitus = pm.Data('tinnitus', df_ecg['tinnitus'].values)

        y_s = pm.Data('y_s', df_ecg[feature].values)

        # Design matrix: [age, tinnitus, age * tinnitus]
        X = pt.stack([age, tinnitus, age * tinnitus], axis=0)

        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=.5, dims='effect')

        sigma = pm.Exponential('sigma', lam=1)
        nu = pm.Gamma('nu', 2, 0.1)
        y = pm.StudentT('y',
                        mu=alpha + pt.dot(beta, X),
                        observed=y_s,
                        nu=nu,
                        sigma=sigma,
                        ) 

        mdf =  pm.sample(**sample_kwargs)

    ch_effects = az.summary(mdf, 
                var_names=['alpha', 'beta'], 
                hdi_prob=.89)

    ch_effects['feature'] = feature

    eff_dfs.append(ch_effects)


df_stats = pd.concat(eff_dfs)


df_stats.to_csv('df_ecg_stats.csv')
# %%
df_stats
# %%
