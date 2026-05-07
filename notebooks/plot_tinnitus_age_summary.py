#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.special as sp

sns.set_theme(context='poster',
              style='ticks')

# %%
thresh = 2.0

df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv', index_col = 0)
df_pe = pd.read_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params__peak_threshold_{thresh}.csv', index_col = 0)


df_data = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])

# %%
data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_tinn_final_final')

cur_feat = 'tau' # Exponent_1, Exponent_2, tau, alpha_cf

if cur_feat == 'Exponent_2':    
    thresh = 3.0 #only refers to oscillatory peaks
else:
    thresh = 2.0

if 'alpha' in cur_feat:
    if cur_feat == 'alpha_osc':
        data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_tinn_final_rkf')

    df = pd.read_csv(data_f / f'{cur_feat}_hi__peak_threshold_{thresh}.csv', index_col = 0)
else:
    df = pd.read_csv(data_f / f'{cur_feat}_hi__peak_threshold_{thresh}.csv', index_col = 0)


interaction = np.array([1 if 'age:tinnitus' in i else 0 for i in df.index])  == 1
age = np.array([1 if 'age' in i else 0 for i in df.index]) - interaction.copy() == 1
tinnitus = np.array([1 if 'tinnitus' in i else 0 for i in df.index]) - interaction.copy() == 1


df_inter = df.iloc[interaction].copy().reset_index()
df_inter['ch_name'] = [i[8:-15] for i in df_inter['index']]
df_age = df.iloc[age].copy().reset_index()
df_age['ch_name'] = [i[8:-6] for i in df_age['index']]
df_tinnitus = df.iloc[tinnitus].copy().reset_index()
df_tinnitus['ch_name'] = [i[8:-11] for i in df_tinnitus['index']]

masks = []
for model in ['tinnitus', 'age', 'inter']: 

    if model == 'tinnitus':
        cur_df = df_tinnitus.copy()
    elif model == 'inter':
        cur_df = df_inter.copy()
    elif model == 'age':
        cur_df = df_age.copy()

    if cur_feat == 'alpha_osc':
            log_rope = 0.05 * sp.constants.pi / np.sqrt(3)
            rope_l, rope_h = -1*log_rope, log_rope #for logistic model 0.1 * pi / np.sqrt(3)
    else:
            rope_l, rope_h = -.05, .05 #-.05, .05
    #
    mask_neg = np.logical_and(cur_df['hdi_94.5%'] < rope_l, cur_df['hdi_5.5%'] < rope_l)
    mask_pos = np.logical_and(cur_df['hdi_94.5%'] > rope_h, cur_df['hdi_5.5%'] > rope_h)

    mask = (mask_pos + mask_neg).astype(int)
    masks.append(mask)

df_mask = pd.DataFrame(np.array(masks).T, columns=['tinnitus', 'age', 'inter'])
df_mask['ch_name'] = cur_df['ch_name']
df_mask['interxage'] = df_mask['inter'] * df_mask['age']
df_mask['interxagextinnitus'] = df_mask['inter'] * df_mask['age'] * df_mask['tinnitus']
# %%
if cur_feat == 'alpha_cf':
     chs2pick = df_mask.query('age > 0')['ch_name'].to_list()
# elif cur_feat == 'Exponent_2':
#      chs2pick = df_mask.query('interxagextinnitus > 0')['ch_name'].to_list()
elif cur_feat in ['tau', 'Exponent_1', 'Exponent_2']:
    chs2pick = df_mask.query('inter > 0')['ch_name'].to_list()
# %%
#chs2pick = df_mask.query('age > 0')['ch_name'].to_list()
# %%
df2plot = df_data.query('ch_name == @chs2pick').groupby('subject_id').mean(numeric_only=True)
# %%
print(chs2pick)

#%%
import scipy.stats as st

g = sns.lmplot(df2plot, x='age', y=cur_feat, hue='tinnitus', aspect=1.15)
sns.despine()
g.figure.savefig(f'../results/age_x_tinnitus_eff_sig_cluster_{cur_feat}.svg')

st.pearsonr(df2plot['age'], df2plot[cur_feat])

# %%
# %%
df2plot = df_data.query('ch_name == @chs2pick')

import scipy.stats as st

g = sns.lmplot(df2plot, x='age', 
               y=cur_feat, 
               hue='tinnitus', 
               col='ch_name', 
               sharey=False,
               col_wrap=5, aspect=1.15)
sns.despine()
#g.figure.savefig(f'../results/age_x_tinnitus_eff_sig_cluster_{cur_feat}.svg')
# %%
