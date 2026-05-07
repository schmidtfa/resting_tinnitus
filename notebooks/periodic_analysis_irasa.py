#%%
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

#%%
#INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/pyrasa_peak_params'
INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/pyrasa_peak_params_duration_6_final_default'

ctx_periodic, peak_list = [], []
delta, theta, alpha, beta = [], [], [], []
bad_subjects = []

thresh = 2.0

for subject_id in subject_ids:

    try:
        cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__peak_threshold_{thresh}.dat'))[0]))

        ctx_periodic.append(cur_data['peaks'].query('cf < 46'))
        peak_list.append(cur_data['df_peaks'])

        delta.append(cur_data['bands']['delta'].sort_values('pw', ascending=False).drop_duplicates('ch_name').query('ch_name != "???"'))
        theta.append(cur_data['bands']['theta'].sort_values('pw', ascending=False).drop_duplicates('ch_name').query('ch_name != "???"'))
        alpha.append(cur_data['bands']['alpha'].sort_values('pw', ascending=False).drop_duplicates('ch_name').query('ch_name != "???"'))
        beta.append(cur_data['bands']['beta'].sort_values('pw', ascending=False).drop_duplicates('ch_name').query('ch_name != "???"'))

    except IndexError:
        print(f'No data for {subject_id}')
        bad_subjects.append(subject_id)

# %%
df_ctx = pd.concat(ctx_periodic)

# %%
no_tinn = df_ctx.query('tinnitus == False')

f, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_ctx, x='cf', hue='tinnitus', #bins=100, 
            palette='deep')
ax.set_xlim(1, 40)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('N peaks')
sns.despine()
f.savefig('../results/all_peaks_hist_tinn_no_tinn.svg')

#%%
f, ax = plt.subplots(figsize=(5, 5))
ax.hist(no_tinn['cf'], bins=40)
ax.set_xlim(1, 40)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('N peaks')
sns.despine()


# %% impute nans
def merge_oscillatory_data(label_list, data_list):

    for ix1, (freq_label, freq) in enumerate(zip(label_list, data_list)):

        df_cur_freq = pd.concat(freq).reset_index()

        ave_freq_ch_df = df_cur_freq.groupby('ch_name')['cf'].mean()
        ave_cf = ave_freq_ch_df.mean()
        ave_freq_ch = ave_freq_ch_df.to_dict()
        #replace alpha center frequency by ave cf
        new_cf_vector = []
        for _, cur in df_cur_freq.iterrows():
            if np.isnan(cur['cf']):
                if np.isnan(ave_freq_ch[cur['ch_name']]) == False:
                    new_cf_vector.append(ave_freq_ch[cur['ch_name']])
                else:
                    new_cf_vector.append(ave_cf)
            else:
                new_cf_vector.append(cur['cf'])

        df_cur_freq[freq_label + '_osc'] = np.isnan(df_cur_freq['cf']) == False
        #df_cur_freq['cf'] = df_cur_freq#np.array(new_cf_vector)
        #df_cur_freq['pw'] = df_cur_freq['pw'].fillna(0)
        #df_cur_freq['bw'] = df_cur_freq['bw'].fillna(0)

        df_cur_freq.rename(columns={'cf': freq_label + '_cf',
                                    'pw': freq_label + '_pw',
                                    'bw': freq_label + '_bw'}, inplace=True)
        df_cur_freq.drop(columns='index', inplace=True)
        if ix1 == 0:
            df2merge = df_cur_freq.copy()
        else:
            df2merge = df2merge.merge(df_cur_freq, on=['ch_name', 'subject_id', 'tinnitus', 'dB', 'age', 'tinnitus_distress'])

    return df2merge
#%%
label_list = ['delta', 'theta', 'alpha', 'beta']
data_list = [delta, theta, alpha, beta]

df_ctx = merge_oscillatory_data(label_list, data_list).query('ch_name != "???"')
df_peaks = pd.concat(peak_list).query('ch_name != "???"')
df_periodic = df_ctx.merge(df_peaks, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])

#%
df_periodic.to_csv(f'../data/periodic_params__peak_threshold_{thresh}.csv')
#%

# param = 'theta_cf'

# f, ax = plt.subplots(figsize=(5, 75))
# sns.pointplot(df_periodic, x=param, y='ch_name', hue='tinnitus', ax=ax)
#%

# med_age = np.median(df_ctx.drop_duplicates(subset='subject_id')['age'])
# df_ctx_o = df_ctx.query('age > 50')
#%

# f, ax = plt.subplots(figsize=(5, 75))
# sns.pointplot(data=df_peaks, y='ch_name', x='n_peaks', hue='tinnitus', ax=ax)

#%%
data2corr = df_periodic.groupby('ch_name')[['alpha_cf', 'age']].corr()
cur_corr = data2corr['age'].copy().reset_index().query(f'level_1 == "alpha_cf"')
plt.hist(cur_corr['age'])

#%%
data2corr = df_periodic.query('tinnitus == True').groupby('ch_name')[['alpha_cf', 'age']].corr()
cur_corr_t = data2corr['age'].copy().reset_index().query(f'level_1 == "alpha_cf"')
plt.hist(cur_corr_t['age'])


#%%
data2corr = df_periodic.query('tinnitus == False').groupby('ch_name')[['alpha_cf', 'age']].corr()
cur_corr_nt = data2corr['age'].copy().reset_index().query(f'level_1 == "alpha_cf"')
plt.hist(cur_corr_nt['age'])

# %
# sns.set_context('paper')
# f, ax = plt.subplots(figsize=(5, 75))
# sns.pointplot(data=df_ctx_o, y='ch_name', x='alpha_cf', hue='tinnitus', ax=ax)

#%%
(cur_corr_t < cur_corr_nt).mean()


# %%
key = 'alpha_pw'
data2corr = df_ctx.groupby('ch_name')[[key, 'age']].corr()
cur_corr = data2corr['age'].copy().reset_index().query(f'level_1 == "{key}"')
plt.hist(cur_corr['age'])

# # %%
# f, ax = plt.subplots(figsize=(5, 75))
# sns.pointplot(data=cur_corr, y='ch_name', x='age', ax=ax)

# %%
