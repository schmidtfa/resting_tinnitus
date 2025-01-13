#%%
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('ticks')
sns.set_context('talk')


# %%

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

df_list = []

for subject_id in subject_ids: 

    try: 
        query_string = '__duration_4__hmax_2__fft_method_irasa__source_surface__atlas_glasser.dat'
        INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'
        cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}' + query_string))[0]))

        cur_s = df_all.query(f'subject_id == "{subject_id}"')

        cur_df = pd.DataFrame({'periodic': cur_data['src']['irasa_label_stc'].periodic.mean(axis=0),
                                    'freqs': cur_data['src']['irasa_label_stc'].freqs,})
        
        cur_df['tinnitus'] = int(cur_s['tinnitus'])
        df_list.append(cur_df)

    except IndexError:
        print(f'No data for {subject_id}') 
# %%
df_cmb = pd.concat(df_list)
df_cmb['abs_p'] = df_cmb['periodic'].abs()
# %%
sns.lineplot(df_cmb.query('freqs < 20'), x='freqs', y='periodic', hue='tinnitus')
# %%
sns.lineplot(df_cmb.query('freqs < 20'), x='freqs', y='abs_p', hue='tinnitus')
# %%
