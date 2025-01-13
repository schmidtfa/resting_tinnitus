#%%
from pathlib import Path
import joblib
import pandas as pd
import pymc as pm
import arviz as az

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')


# %%
INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
all_files = list(Path(INDIR).glob('*/*.dat'))

# %%
all_data = []

for f in all_files:
    cur_data = joblib.load(f)
    # %
    cols_of_interest = ['subject_id', 'tinnitus', 'dB', 'gender', 
                        'tinnitus_distress', 'measurement_age']

    df_demographics = cur_data['subject_info'][cols_of_interest]

    df_physio = pd.DataFrame({'EOGV': cur_data['eog'][0,:],
                'EOGH': cur_data['eog'][1,:],
                'ECG': cur_data['ecg'][0,:],
                'MEG': cur_data['src']['label_tc'].mean(axis=0),
                'Frequency': cur_data['freq'],
                'subject_id': cur_data['subject_id'],
                }) 
    physio_melt = df_physio.melt(id_vars=['Frequency', 'subject_id'], var_name='channel', value_name='Power (a.u.)')

    df_physio_merge = physio_melt.merge(df_demographics, on='subject_id')
    all_data.append(df_physio_merge)
# %%
df_cmb = pd.concat(all_data)
# %%
df_cut = (df_cmb.query('Frequency < 40')
                .query('channel == "MEG"')
                .query('subject_id != @bad_subjects'))

#%%
fig, ax = plt.subplots(figsize=(5,5))
sns.lineplot(df_cut.query('tinnitus == False'), x='Frequency' ,y='Power (a.u.)', ax=ax).set(xscale='log', yscale='log')


# %%
g = sns.FacetGrid(data=df_cut, sharey=False, col='channel', col_wrap=2, hue='tinnitus')
g.map(sns.lineplot, 'Frequency', 'Power (a.u.)',)#.set(xscale='log', yscale='log')

# %%
