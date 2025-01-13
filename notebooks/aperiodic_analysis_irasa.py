#%%
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

#%%
#INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/pyrasa_ap_params'
INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/pyrasa_ap_params_duration_6'

fixed_chs, knee_chs = [], []
bad_subjects = []
ic='BIC'
aperiodic_data, aperiodic_data_sc = [], []

for subject_id in subject_ids:

    try:
        cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__cur_ic_{ic}.dat'))[0]))

        fixed_chs.append(cur_data['ctx']['ch_coice']['fixed'])
        knee_chs.append(cur_data['ctx']['ch_coice']['knee'])

        aperiodic_data.append(cur_data['ctx']['aperiodic_params']['knee'])

    except IndexError:
        print(f'No data for {subject_id}')
        bad_subjects.append(subject_id)
# %%
n_subs = len(subject_ids) - len(bad_subjects)

ch_labels, counts = np.unique(np.concatenate(fixed_chs), return_counts=True)

df_chs = pd.DataFrame({'ch_name': ch_labels, 
                       'counts': counts},)
df_chs['proba_fixed'] = df_chs['counts'] / n_subs
# %%
plt.hist(df_chs['proba_fixed'])

# %%
df_ap = pd.concat(aperiodic_data)
df_ap_o = df_ap.query('age > 50')

df_ap.query('ch_name != "???"').to_csv('../data/aperiodic_params.csv')
#%%
param = 'Exponent_2'

f, ax = plt.subplots(figsize=(5, 75))
sns.pointplot(df_ap, x=param, y='ch_name', hue='tinnitus', ax=ax)

# %%

import scipy.stats as stats
list_k =['Offset', 'Exponent_1', 'Exponent_2', 'Knee Frequency (Hz)', 'tau', 'age', 'tinnitus', 'dB']
list_f =['Offset', 'Exponent', 'age', 'tinnitus']

data2corr = df_ap.groupby('subject_id')[list_k].mean()

#%%
data2corr.corr('pearson')

#%%
sns.pointplot(data2corr, x='tinnitus', y='Exponent_2', errorbar='se')

# %%
data2corr = df_ap.groupby('ch_name')[list_k].corr()

# %%
data2corr
# %%
key = 'tau'
cur_corr = data2corr['age'].copy().reset_index().query(f'level_1 == "{key}"')
# %%
plt.hist(cur_corr['age'])
# %%
