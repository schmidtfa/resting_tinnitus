#%%
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import zscore

from ggseg_py.ggseg_py import rda2gpd, merge_data
from ggseg_py.plotting_utils import plot_surface
from pathlib import Path

data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_cc_final_final')

cur_feat = 'alpha_bw' # Exponent_1, Exponent_2, tau, alpha_cf
rope = True

df = pd.read_csv(data_f / f'{cur_feat}_hi.csv', index_col = 0).reset_index()



#interaction = np.array([1 if 'age:tinnitus' in i else 0 for i in df.index])  == 1
#age = np.array([1 if 'age' in i else 0 for i in df.index]) - interaction.copy() == 1
#tinnitus = np.array([1 if 'tinnitus' in i else 0 for i in df.index]) - interaction.copy() == 1

df['roi'] = [i[8:-1] for i in df['index']]

if rope:
    rope_l, rope_h = -.05, .05
else:
    rope_l, rope_h = 0, 0
#


mask_neg = np.logical_and(df['hdi_94.5%'] < rope_l, df['hdi_5.5%'] < rope_l)
mask_pos = np.logical_and(df['hdi_94.5%'] > rope_h, df['hdi_5.5%'] > rope_h)

mask = np.logical_or(mask_pos, mask_neg)


df['mean_mask'] = df['mean'] * mask

gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
              atlas_name= 'glasser')
df_mg = gdf.merge(df, on='roi')

if np.logical_and(np.sum(mask_pos) > 0, np.sum(mask_neg) > 0):
    cmap = 'RdBu_r'
    vmin, vmax = -.5, .5
elif np.sum(mask_pos) > 0:
    cmap = 'Reds'
    vmin, vmax = 0, .5
elif np.sum(mask_neg) > 0:
    cmap = 'Blues_r'
    vmin, vmax = -0.5, 0
else:
    cmap = 'RdBu_r'
    vmin, vmax = -.5, .5


f, ax = plot_surface(df_mg,
                    column='mean_mask',
                    cmap=cmap, 
                    show_cbar=True,
                    vmin=vmin,
                    vmax=vmax
                    )


f.savefig(f'../results/final_shit/{cur_feat}_cc_rope_{rope}.svg')
# %%
df.loc[mask == 1]['mean'].mean()

#%%
df.loc[mask == 1]['mean'].min()

#%%
df.loc[mask == 1]['mean'].max()
# %%

df.loc[mask_neg == 1]['mean_mask'].mean()
# %%
#%%
df.loc[mask_neg == 1]['mean_mask'].min()

#%%
df.loc[mask_neg == 1]['mean_mask'].max()
# %%
