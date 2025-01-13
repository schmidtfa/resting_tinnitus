#%%
import joblib
from plus_slurm import Job
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


import numpy as np
import pandas as pd

#%%
class RandomForest(Job):

    def run(self,
            cur_ch,
            n_splits=5,
            n_repeats=10,
            ):

        #%%
        df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
        df_pe = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params.csv')

        df_cmb = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])
        # #% debug
       #  all_chs = df_cmb['ch_name'].unique()
       #  cur_ch = all_chs[0]
       #  n_splits=5
       #  n_repeats=5

        #%%
        #df_cmb = df_cmb.query('age > 50')
        #df_cmb['old'] = df_cmb['age'] > 50 #median age is 60
        cur_ch_df = df_cmb.query(f'ch_name == "{cur_ch}"').reset_index()
        cur_ch_df.drop_duplicates(subset='subject_id', inplace=True)


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

        #np.random.shuffle(cur_ch_df['tinnitus'])
        y = cur_ch_df['tinnitus']
        X = cur_ch_df[predictors]

        #%%
        importances_rf, scores, settings = [], [], []
        for ix in range(n_repeats):
        
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ix)

                for ix_nested, (train_ix, test_ix) in enumerate(skf.split(X, y)):

                        print(f'Running Nested CV on split {ix_nested+1} of {n_splits*n_repeats}')
                        
                        cur_train_x = X.iloc[train_ix]
                        cur_train_y = y.iloc[train_ix]

                        cur_test_x = X.iloc[test_ix]
                        cur_test_y = y.iloc[test_ix]

                        skf_nested = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ix_nested)
                        #skf_nested = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=ix_nested)

                        param_grid = {
                                'n_estimators': [100, 200, 500],  # Number of trees in the forest
                                'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
                                'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                                'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
                                }


                        clf_cv = GridSearchCV(RandomForestClassifier(),
                                                param_grid=param_grid,
                                                cv=skf_nested,
                                                scoring='roc_auc',
                                                refit=True,
                                                n_jobs=-1
                                                )
                        
                        clf_cv.fit(cur_train_x, cur_train_y) 
                        #get settings to check later
                        settings.append(pd.DataFrame(clf_cv.best_params_, index=[ix]))
                        
                        #Calculate feature importance
                        # importance_perm = permutation_importance(clf_cv, cur_test_x, cur_test_y, 
                        #                                          n_repeats=n_repeats, random_state=ix, n_jobs=-1)

                        # importances_perm_tmp = pd.DataFrame(importance_perm.importances.T,
                        #                                         columns=predictors)
                        # importances_perm_tmp['ch_name'] = cur_ch
                        # importances_perm.append(importances_perm_tmp)


                        importances_rf.append(pd.DataFrame({'feature': predictors,
                                                           'ch_name': cur_ch,
                                                           'importance_rf': clf_cv.best_estimator_.feature_importances_,
                                                               }))

                        scores.append(clf_cv.score(cur_test_x, cur_test_y)) #

    
        df_acc = pd.DataFrame({'ch_name': cur_ch,
                               'accuracy (%)': np.array(scores) * 100,
                               })

        df_importance_rf = pd.concat(importances_rf)
        #df_importance_perm = pd.concat(importances_perm)

        df_settings = pd.concat(settings)
        df_settings['ch_name'] = cur_ch

        data = {'acc': df_acc,
                'importance_rf': df_importance_rf,
                #'importance_perm': df_importance_perm,
                'settings': df_settings}


        out_f = f'/home/schmidtfa/experiments/resting_tinnitus/data/random_forest/{cur_ch}.dat'

        joblib.dump(data, out_f)
# %%
