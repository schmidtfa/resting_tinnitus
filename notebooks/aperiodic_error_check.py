#%%
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()


INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'

error_list = []
for cur_hmax in [2, 3, 4]:
    query_string = f'__duration_2__hmax_{cur_hmax}__source_surface__atlas_glasser.dat'
    for subject_id in subject_ids:

        try:
            cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}' + query_string))[0]))
            cur_s_error = cur_data['src']['aperiodic_error'].mean(axis=1)
            error_list.append(pd.DataFrame({'error': cur_s_error,
                                     'ch_labels':  cur_data['src']['label_info']['names_order_mne'],
                                     'hmax': cur_hmax,
                                     'subject_id': subject_id}))

        except IndexError:
            print(f'No data for {subject_id}')
# %%
error_df = pd.concat(error_list)

# %%
f, ax = plt.subplots(figsize=(5, 70))
sns.pointplot(error_df, y='ch_labels', x='error', hue='hmax', ax=ax)
sns.despine()
# %%
