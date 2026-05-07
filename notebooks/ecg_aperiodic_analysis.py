#%%
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

duration = 6
#%%
query_string = f'__duration_{duration}__hmax_2__source_surface__atlas_glasser.dat'
INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'

fixed_chs, knee_chs = [], []
bad_subjects = []
ic='BIC'
aperiodic_data, aperiodic_data_sc = [], []

knee_dfs = []
for subject_id in subject_ids:

    try:
        cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}' + query_string))[0]))

        def add_info(df, df_info):
            df['subject_id'] = df_info['subject_id'].iloc[0]
            df['tinnitus'] = df_info['tinnitus'].iloc[0]
            df['dB'] = df_info['dB'].iloc[0]
            df['age'] = df_info['measurement_age'].iloc[0]
            df['tinnitus_distress'] = df_info['tinnitus_distress'].iloc[0]

            return df
        
        knee_model = cur_data['ecg'].fit_aperiodic_model(fit_func='knee')

        df_knee = add_info(knee_model.aperiodic_params, cur_data['subject_info'])

        knee_dfs.append(df_knee)
    except IndexError:
        print(f'No data for {subject_id}')
        bad_subjects.append(subject_id)
# %%

pd.concat(knee_dfs).to_csv('ecg_aperiodic_fit.csv')

# %%
