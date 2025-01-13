#%%
import pandas as pd
import pymc_bart as pmb
import pymc as pm
from os.path import join, isdir
import os
from sklearn.model_selection import StratifiedKFold
from plus_slurm import Job


# %%
class BART(Job):

    def run(self,
            cur_ch,
            n_trees=20,
            n_splits=5,
            ):

        #%%
        df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
        df_pe = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params.csv')

        df_cmb = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])
        #% debug
        all_chs = df_cmb['ch_name'].unique()
        cur_ch = all_chs[0]
        n_trees=20
        n_splits=5

        df_cmb['old'] = df_cmb['age'] > 50 #median age is 60
        cur_ch_df = df_cmb.query(f'ch_name == "{cur_ch}"').reset_index()
        cur_ch_df.drop_duplicates(subset='subject_id', inplace=True)

        cur_ch_df = df_cmb.query(f'ch_name == "{cur_ch}"').reset_index()

        predictors = ['Offset', 
                      'Exponent_1',  
                      'Exponent_2',  
                 #     'Knee Frequency (Hz)', 
                      'tau',                  
                #      'n_peaks', 
                #      'delta_cf', 'delta_pw',#'theta_bw',
                #      'theta_cf', 'theta_pw',#'theta_bw',
                #       #'alpha_osc',
                       'alpha_cf', 'alpha_pw', #'alpha_bw',
                #       #'beta_osc', 
                #       'beta_cf', 'beta_pw', #'beta_bw', 
                    ]

        y = cur_ch_df['tinnitus']
        X = cur_ch_df[predictors]

        #%%
        sample_kwargs = {
                        'draws': 1000,
                        'tune': 1000,
                        'chains': 4,
                        'idata_kwargs' : {'log_likelihood': True}, 
                        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for ix, (train_ix, test_ix) in enumerate(skf.split(X, y)):
            
            cur_train_x = X.iloc[train_ix]
            cur_train_y = y.iloc[train_ix]

            cur_test_x = X.iloc[test_ix]
            cur_test_y = y.iloc[test_ix]

            with pm.Model() as model:
                # data containers
                model.add_coord('id', cur_train_x.index, mutable=True)
                model.add_coord('feature', cur_train_x.columns, mutable=True)
                
                X_s = pm.Data('X_s', cur_train_x, dims=('id', 'feature'))
                y_s = pm.Data("y_s", cur_train_y, dims='id')
                
                # model definiton
                mu = pmb.BART("mu", X=X_s, Y=cur_train_y, m=n_trees, dims='id')
                # link function
                p = pm.Deterministic("p", pm.math.invlogit(mu))        
                # likelihood
                y_pred = pm.Bernoulli("y_pred", p=p, observed=y_s, dims='id')

            #% training
            with model:
                # actual training via MCMC
                idata = pm.sample(**sample_kwargs)

            #% testing
            with model:
                pm.set_data({"X_s": cur_test_x, 
                             "y_s": cur_test_y}, coords={'id': cur_test_x.index})
                idata.extend(pm.sample_posterior_predictive(idata))
                
            #add test data for post-hoc comparison
            idata.observed_data['y_test_set'] = cur_test_y
            outdir = f'/home/schmidtfa/experiments/resting_tinnitus/data/bart/n_trees_{n_trees}'

            if not isdir(outdir):
                os.makedirs(outdir)

            idata.to_netcdf(join(outdir, f'{cur_ch}_fold_{ix}.nc'))
# %%
