#%%
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import scipy as sp


from scipy.stats import zscore

from ggseg_py.ggseg_py import rda2gpd, merge_data
from ggseg_py.plotting_utils import plot_surface
from pathlib import Path

data_f = Path('/home/schmidtfa/experiments/resting_tinnitus/data/age_tinn_final_final_mosi')

cur_feat = 'Exponent_2' # Exponent_1, Exponent_2, tau, alpha_cf
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
df_inter['roi'] = [i[8:-15] for i in df_inter['index']]
df_age = df.iloc[age].copy().reset_index()
df_age['roi'] = [i[8:-6] for i in df_age['index']]
df_tinnitus = df.iloc[tinnitus].copy().reset_index()
df_tinnitus['roi'] = [i[8:-11] for i in df_tinnitus['index']]


for model in ['tinnitus', 'age', 'inter']: 
#model = 'inter'
    for rope in [True, False]:


        if model == 'tinnitus':
            cur_df = df_tinnitus.copy()
        elif model == 'inter':
            cur_df = df_inter.copy()
        elif model == 'age':
            cur_df = df_age.copy()

        if rope:
            if cur_feat == 'alpha_osc':
                log_rope = 0.05 * sp.constants.pi / np.sqrt(3)
                rope_l, rope_h = -1*log_rope, log_rope #for logistic model 0.1 * pi / np.sqrt(3)
            else:
                rope_l, rope_h = -.05, .05

        else:
            rope_l, rope_h = -.0, .0
        #
        mask_neg = np.logical_and(cur_df['hdi_94.5%'] < rope_l, cur_df['hdi_5.5%'] < rope_l)
        mask_pos = np.logical_and(cur_df['hdi_94.5%'] > rope_h, cur_df['hdi_5.5%'] > rope_h)

        mask = (mask_pos + mask_neg).astype(int)

        if cur_feat == 'alpha_osc':
            cur_df['mean_mask'] = np.exp(cur_df['mean'])
            cur_df['mean_mask'][mask == 0] = 0
            cur_df['mean_mask'][cur_df['mean_mask'] == 0] = 1

        else:
            cur_df['mean_mask'] = cur_df['mean']
            cur_df['mean_mask'][mask == 0] = 0

        gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
                    atlas_name= 'glasser')
        df_mg = gdf.merge(cur_df, on='roi')

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

        if cur_feat == 'alpha_osc': 
                cmap = 'RdBu_r'
                vmin, vmax = .5, 2#cur_df['mean_mask'].min(), cur_df['mean_mask'].max() + 0.01

        f, axes = plot_surface(df_mg,
                            column='mean_mask',
                            cmap=cmap, 
                            show_cbar=False,
                            vmin=vmin,
                            vmax=vmax
                            )

        if cur_feat == 'alpha_osc':
            import matplotlib.cm as cm
            from matplotlib.colors import LogNorm

            # after you already did:
            # fig, axes = plot_surface(...)

            norm = LogNorm(vmin=vmin, vmax=vmax)

            # 1) update the patch collections in each axis
            for ax in axes.ravel():
                for coll in ax.collections:
                    arr = coll.get_array() if hasattr(coll, "get_array") else None
                    if arr is None:
                        continue
                    coll.set_norm(norm)
                    coll.set_clim(vmin, vmax)   # optional, but fine
                    coll.changed()          # nudge redraw

            # 2) add (or replace) a colorbar that matches
            # (ggseg_py doesn't return the cbar handle, so easiest is: show_cbar=False, then add your own)
            sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r"))
            sm.set_array([])  # helps some mpl versions

            cbar = f.colorbar(sm, ax=axes, fraction=0.05, pad=0.02)
            cbar.set_ticks([vmin, 1, vmax])

            f.canvas.draw_idle()

        #f.savefig(f'../results/final_shit/{cur_feat}_{model}_rope_{rope}.svg')


# %%

cur_df.loc[mask == 1]['mean'].mean()
# %%
#%%
cur_df.loc[mask == 1]['mean'].min()

#%%
cur_df.loc[mask == 1]['mean'].max()
# %%

cur_df.loc[mask_neg == 1]['mean_mask'].mean()

# %%
#%%
cur_df.loc[mask_neg == 1]['mean_mask'].min()

#%%
cur_df.loc[mask_neg == 1]['mean_mask'].max()
# %%
