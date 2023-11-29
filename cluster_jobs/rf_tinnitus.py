#%%
import joblib
from plus_slurm import Job
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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
        df_cmb = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_all_spec_features.csv')
        # #% debug
        # all_chs = df_cmb['ch_name'].unique()
        # cur_ch = all_chs[0]
        # n_splits=5
        # n_repeats=5
        # tn_match = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')[['measurement_age', 'subject_id']]

        #%%
        cur_ch_df = df_cmb.query(f'ch_name == "{cur_ch}"').reset_index()
        #cur_ch_df = cur_ch_df.merge(tn_match, on='subject_id').query('measurement_age > 40')

        predictors = ['offset', 
                      'exponent',                     
                #       'n_peaks', 
                       'theta_cf', 'theta_pw',#'theta_bw',
                #       #'alpha_osc',
                       'alpha_cf', 'alpha_pw', #'alpha_bw',
                #       #'beta_osc', 
                       'beta_cf', 'beta_pw', #'beta_bw',
                #       #'gamma_osc',
                       'gamma_cf', 'gamma_pw',#'gamma_bw', 
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

                        params = {'n_estimators': [100, 200, 300],
                                  #'min_samples_leaf': [4, 6, 10],
                                  #'C': [0.1, 0.25, .5, 1],
                                 }

                        clf_cv = GridSearchCV(RandomForestClassifier(),
                                                param_grid=params,
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


        out_f = f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/ada_boost/{cur_ch}.dat'

        joblib.dump(data, out_f)
# %%
