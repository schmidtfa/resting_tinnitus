#%%
import pandas as pd
from pathlib import Path
import joblib
import pymc_bart as pmb
import pymc as pm
import arviz as az
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split, StratifiedKFold



#%%
df_cmb = pd.read_csv('../data/tinnitus_all_spec_features.csv')

all_chs = df_cmb['ch_name'].unique()

cur_ch = all_chs[0]

#%%
cur_ch_df = df_cmb.query(f'ch_name == "{cur_ch}"').reset_index()

predictors = ['offset', 
              'exponent', 
              #'knee_freq', 
              'n_peaks', 
              'delta_osc',#'delta_cf', 'delta_pw','delta_bw',
              'theta_osc', #'theta_cf', 'theta_pw','theta_bw',
              'alpha_osc', 'alpha_cf', 'alpha_pw', #'alpha_bw',
              'beta_osc', 'beta_cf', 'beta_pw',# 'beta_bw',
              'gamma_osc', #'gamma_cf', 'gamma_pw','gamma_bw', 
              'ch_name', 'subject_id'
              ]

#y = cur_ch_df['tinnitus']
#X = cur_ch_df[predictors]
y = df_cmb[['subject_id', 'ch_name', 'tinnitus']].set_index(['subject_id', 'ch_name'])
X = df_cmb[predictors].set_index(['subject_id', 'ch_name'])

#%% 3D approach (should work but doesnt)
# % do a train test split of the data 
train_ix, test_ix = train_test_split(X.index.levels[0])
X_train, X_test, y_train, y_test = X.loc[train_ix].copy(), X.loc[test_ix].copy(), y.loc[train_ix].copy(), y.loc[test_ix].copy()

#%%
sample_kwargs = {
                'draws': 1000,
                'tune': 1000,
                'chains': 4,
                }

s_ixs4train = np.arange(len(train_ix))
ch_ixs4train = np.arange(360)

#%%

train_y_np = np.squeeze(y_train.to_xarray().to_array().to_numpy())#.swapaxes(0,1)
train_x_np = X_train.to_xarray().to_array().to_numpy()#.swapaxes(0,1)

with pm.Model() as model:
    # Coordinates
    model.add_coord('s_id', train_ix, mutable=True)
    model.add_coord('ch_name', X_train.index.levels[1], mutable=True)
    model.add_coord('feature', X_train.columns, mutable=True)
    # data containers
    X_s = pm.MutableData('X_s', np.squeeze(train_x_np[:,:,ch_ixs4train]), dims=('feature', 's_id', 'ch_name'))
    y_s = pm.MutableData("y_s", np.squeeze(train_y_np[:,ch_ixs4train]), dims=('s_id', 'ch_name'))
    # model definiton
    mu = pmb.BART("mu", X=X_s, 
                        Y=y_s,
                        m=2,
                        dims=('s_id', 'ch_name')
                        )
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    #p = pm.Deterministic("p", pm.math.invprobit(mu))
    # likelihood
    y_pred = pm.Bernoulli("y_pred",
                          p=p, 
                          observed=train_y_np, #y_s.T,
                          dims=('s_id', 'ch_name')
                          )

#%% training
with model:
    # actual training via MCMC
    idata = pm.sample(**sample_kwargs)
    
        

#%% 2D approach

train_ix, test_ix = train_test_split(X.index.levels[0], shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = X.loc[train_ix].copy(), X.loc[test_ix].copy(), y.loc[train_ix].copy(), y.loc[test_ix].copy()

X_train.reset_index(inplace=True) 
X_test.reset_index(inplace=True)
y_train.reset_index(inplace=True)
y_test.reset_index(inplace=True)

#%%
sample_kwargs = {
                'draws': 1000,
                'tune': 1000,
                'chains': 4,
                }

s_ixs4train = np.arange(len(train_ix))
ch_ixs4train = np.arange(360)

#%%
ch_ixs, channel = pd.factorize(X_train['ch_name'])
s_ixs, subject = pd.factorize(X_train['subject_id'])

np_x_train = X_train.drop(columns=['subject_id', 'ch_name']).to_numpy()
np_y_train = np.squeeze(y_train.drop(columns=['subject_id', 'ch_name']).to_numpy())

with pm.Model() as model:
    # Coordinates
    model.add_coord('ch_ix', ch_ixs, mutable=True)
    model.add_coord('ch_name', channel, mutable=True)
    model.add_coord('s_ix', s_ixs, mutable=True)
    model.add_coord('subject', subject, mutable=True)
    model.add_coord('feature', X_train.drop(columns=['subject_id', 'ch_name']).columns, mutable=True)


    # data containers
    X_s = pm.MutableData('X_s', np_x_train, dims=('ch_ix', 'feature'))
    y_s = pm.MutableData("y_s", np_y_train, dims='ch_ix')
    # model definiton
    mu = pmb.BART("mu", 
                  X=X_s[ch_ixs], 
                  Y=np_y_train[ch_ixs], 
                  m=2,
                  dims='ch_ix')
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    #p = pm.Deterministic("p", pm.math.invprobit(mu))
    # likelihood
    y_pred = pm.Bernoulli("y_pred",
                          p=p[s_ixs], 
                          observed=y_s,
                          dims='s_ix'
                          )

#%% training
with model:
    # actual training via MCMC
    idata = pm.sample(**sample_kwargs)

# %%
pmb.plot_convergence(idata, var_name="p");


#%% testing
with model:
    pm.set_data({"X_s": X_test, 
                 "y_s": y_test}, coords={'id': X_test.index})
    idata.extend(pm.sample_posterior_predictive(idata))

#%%
az.summary(idata, var_names='p', filter_vars='like')

# %%
pmb.plot_variable_importance(idata, mu, X_train, samples=100);
# %%
y_pred = (idata.posterior_predictive.y_pred.mean(axis=0).mean(axis=0) > 0.5).to_numpy().astype(int)
# %%
(y_test == y_pred).mean()


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer, SlidingEstimator, Scaler
from sklearn.impute import SimpleImputer#,IterativeImputerdata_cmb
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from mne.decoding import cross_val_multiscore

from sklearn.model_selection import cross_val_score

sorted_df = df_cmb.set_index(['subject_id', 'ch_name'])

predictors.remove('subject_id')
predictors.remove('ch_name')

#%%
X = sorted_df[predictors].to_xarray().to_array().to_numpy().swapaxes(0,1)
y = sorted_df['tinnitus'].to_xarray().to_numpy()

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

#constant_imputer = lambda X : np.array([SimpleImputer(strategy='constant', fill_value=0).fit_transform(ch) for ch in X.T]).T
d3_scaler = lambda X : np.array([RobustScaler().fit_transform(sub) for sub in X])


clf = make_pipeline(
                    #FunctionTransformer(d3_scaler),
                    SlidingEstimator(RandomForestClassifier()))

scores = cross_val_multiscore(clf, X_train, y_train, cv=5, scoring='balanced_accuracy')
scores.mean()

#%%
scores.mean(axis=0)
# %%
df_acc = pd.DataFrame({'ch_name': df_cmb['ch_name'].unique(),
                       'acc': scores.mean(axis=0)})
# %%
df_acc[df_acc['acc'] > 0.6]

# %%
mdf = clf.fit(X_train, y_train)

# %%
scores2 = clf.score(X_test, y_test)
# %%
feature_importance = dict()
feature_importance[measure][:,run] = clf.feature_importances_