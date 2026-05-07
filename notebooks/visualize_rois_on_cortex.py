#%%
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.stats import zscore

from ggseg_py.ggseg_py import rda2gpd, merge_data
from ggseg_py.plotting_utils import plot_surface, plot_view

df_regions_info = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/regions_hcmp.csv')
df_regions_info['roi'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
df_regions_info['roi'] = df_regions_info['roi'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                         'R_7Pl_ROI': 'R_7PL_ROI',})

df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']


# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 20 colors from tab20
tab20 = plt.cm.get_cmap('tab20').colors   # shape (20, 4)

# Take 2 extra colors from tab10
tab10 = plt.cm.get_cmap('tab10').colors   # shape (10, 4)
extra = tab10[:2]                          # first 2

# Stack into a single (22, 4) array
colors22 = np.vstack([tab20, extra])

# Build a categorical colormap
cmap22 = ListedColormap(colors22, name="tab22")




gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df_regions_info, on='roi')

# %%
f, ax = plot_surface(df_mg,
             column='cortex',
             cmap=cmap22, 
             show_cbar=True,
             #vmin=0
             )

f.savefig('cortical_regions.svg')
# %%

roi_palette = sns.hls_palette(360,)# s=.4)
from matplotlib.colors import ListedColormap


def make_roi_cmap(n_colors=360,
                  cmap_names=('viridis', 'plasma', 'cividis', 'magma'),
                  seed=0):
    """
    Create a ListedColormap with n_colors, combining several continuous colormaps
    and shuffling the result so adjacent indices are not all similar.
    """
    colors = []

    # how many colors to take from each base cmap
    n_per = int(np.ceil(n_colors / len(cmap_names)))

    for name in cmap_names:
        base = plt.cm.get_cmap(name, n_per)
        colors.extend(base(np.arange(n_per)))

    colors = np.array(colors)[:n_colors]

    # shuffle for visual separation
    rng = np.random.default_rng(seed)
    rng.shuffle(colors)

    return ListedColormap(colors)

# build it
roi_cmap = make_roi_cmap(360)
#%%
df_mg['val'] = 0.95

f, ax = plot_view(df_mg,
             hemi='left',
             side='lateral',
             column='val',
             cmap='Reds', 
             show_cbar=False,
             vmin=0.8,
             vmax=1
             )

f.savefig('cmap_single_cmb.svg')

# %%

from ggseg_py.ggseg_py import rda2gpd
from ggseg_py.plotting_utils import plot_surface

gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df, on='roi')

f, ax = plot_surface(df_mg,
             column='correlation',
             cmap='RdBu_r', 
             show_cbar=True,
             #vmin=0
             )

f.savefig('cortical_regions.svg')