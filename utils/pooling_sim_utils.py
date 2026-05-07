#%%
import numpy as np
from scipy.stats import zscore
import pandas as pd
from collections.abc import Iterable
import numpy as np




def pooling_sim(df_roi,
                cortex_of_interest='Primary_Visual',
                intercept = 0,
                slope_effect = 0.5,
                slope_null_effect = 0.,
                n_subjects = 40,
                seed_value = 42,
                ):


        rng = np.random.default_rng(seed_value)

        subjects = np.arange(n_subjects)
        x = zscore(subjects)
        sigma = 1

        dfs, eff_roi, sim_slopes = [], [], []
        for roi in df_roi['roi']:

            cortex = df_roi.query(f'roi == "{roi}"')['cortex_info'].values[0]

            if cortex_of_interest in cortex:
                
                if isinstance(slope_effect, Iterable) and not isinstance(slope_effect, (str, bytes, bytearray)):
                    cur_slope = rng.choice(slope_effect)
                    y = intercept + cur_slope*x + sigma*rng.normal(0, 1, n_subjects)
                else:
                    cur_slope = slope_effect
                    y = intercept + cur_slope*x + sigma*rng.normal(0, 1, n_subjects)
                
                eff_roi.append(roi) 
                sim_slopes.append(cur_slope)
            else:
                y = intercept + slope_null_effect*x + sigma*rng.normal(0, 1, n_subjects)
                sim_slopes.append(slope_null_effect)

            cur_df = pd.DataFrame({
                        'y': y,
                        'X': x,
                        'subject_id': subjects})
            cur_df['roi'], cur_df['cortex'] = roi, cortex

            dfs.append(cur_df)


        #% combine sim slopes with roi
        df_sim_params = pd.DataFrame({'roi': df_roi['roi'].values, 
                                      'sim_slopes': sim_slopes})

        df_all = pd.concat(dfs)

        dfs = []
        for subject_id in df_all['subject_id'].unique():

            cur = df_all.query(f'subject_id == {subject_id}')

            order = df_roi['roi'].values

            cur_or = (cur.set_index("roi")    
                        .loc[order]         
                        .reset_index()         
            )

            dfs.append(cur_or)

        df_all = pd.concat(dfs)

        return df_all, eff_roi, df_sim_params