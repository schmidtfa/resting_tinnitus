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

df_ap = pd.read_csv('/home/schmidtfa/experiments/brain_age/data/data_cam_can/aperiodic_params.csv', index_col = 0)
df_pe = pd.read_csv('/home/schmidtfa/experiments/brain_age/data/data_cam_can/periodic_params.csv', index_col = 0)


df_data = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'age'])

# %%

cur_feat = 'Exponent_1' # Exponent_1, Exponent_2, tau, alpha_cf


data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_cc_final_final')

df = pd.read_csv(data_f / f'{cur_feat}_hi.csv', index_col = 0).reset_index()


df['roi'] = [i[8:-1] for i in df['index']]


rope_l, rope_h = -.05, .05

mask_neg = np.logical_and(df['hdi_94.5%'] < rope_l, df['hdi_5.5%'] < rope_l)
mask_pos = np.logical_and(df['hdi_94.5%'] > rope_h, df['hdi_5.5%'] > rope_h)

mask = (mask_pos + mask_neg).astype(int)

df_mask = pd.DataFrame(np.array(mask).T, columns=['age'])
df_mask['ch_name'] = df['roi']

chs2pick = df_mask.query('age > 0')['ch_name'].to_list()

# %%
#chs2pick = df_mask.query('age > 0')['ch_name'].to_list()
# %%
df2plot = df_data.query('ch_name == @chs2pick').groupby('subject_id').mean(numeric_only=True)
# %%

import scipy.stats as st
f, ax = plt.subplots(figsize=(5,4))
sns.regplot(df2plot, x='age', y=cur_feat, ax=ax,
    scatter_kws={"color": "#666666", "alpha": 0.6, #"s":60, 
                 'alpha':.5},
    line_kws={"color": sns.color_palette('deep')[3], "linewidth": 2})
ax.set_xlabel('age (years)')
sns.despine()



f.savefig(f'../results/age_x_camcan_eff_sig_cluster_{cur_feat}.svg')

st.pearsonr(df2plot['age'], df2plot[cur_feat])

# %%

# %%
