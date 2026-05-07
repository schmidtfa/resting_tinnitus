#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ggseg_py.ggseg_py import rda2gpd, merge_data
from ggseg_py.plotting_utils import plot_surface

from pathlib import Path

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_theme(style='ticks',
              context='poster',
              palette='deep')

cmap = sns.color_palette('deep').as_hex()

tin_c = cmap[1]
no_tin_c = cmap[0]
thresh = 2.0
#%%
out_p = Path('/home/schmidtfa/experiments/resting_tinnitus/results/final_shit/descriptives')

df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
df_pe = pd.read_csv(f'/home/schmidtfa/experiments/resting_tinnitus/data/periodic_params__peak_threshold_{thresh}.csv')

df = df_ap.merge(df_pe, on=['ch_name', 'subject_id', 'tinnitus','dB', 'age', 'tinnitus_distress'])
df = df[df['Knee Frequency (Hz)'] < 100]
df['Exponent_1_log'] = np.log(df['Exponent_1'])
df['tau_log'] = np.log(df['tau'])

df_nt = df.query('tinnitus == False')
# %%
n_bins = 100
f, ax = plt.subplots(figsize=(5,5))

sns.histplot(data=df, x='Offset', hue='tinnitus', hue_order=[False, True],
             bins=n_bins, ax=ax)
ax.set_xlim(0, 5)
sns.despine()
#f.tight_layout()
f.savefig(out_p / 'offset_hist.svg')
# %%
f, ax = plt.subplots(figsize=(5,5))
sns.histplot(data=df, x='Exponent_1', hue='tinnitus', 
             hue_order=[False, True],
             ax=ax, bins=n_bins, log_scale=True)
sns.despine()
f.savefig(out_p / 'Exponent_1_hist.svg')
# %%
f, ax = plt.subplots(figsize=(5,5))
sns.histplot(data=df, x='Exponent_2', hue='tinnitus', 
hue_order=[False, True],
             ax=ax, bins=n_bins)
sns.despine()
f.savefig(out_p / 'Exponent_2_hist.svg')

#%%
f, ax = plt.subplots(figsize=(5,5))
sns.histplot(data=df, x='tau', hue='tinnitus',
             hue_order=[False, True], 
             ax=ax, bins=n_bins, log_scale=True)
sns.despine()
f.savefig(out_p / 'tau_hist.svg')


# %%

df_nt.rename({'ch_name': 'roi'}, axis=1, inplace=True)

df_avf = df_nt.groupby('roi').mean(numeric_only=True).reset_index()
gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
              atlas_name= 'glasser')
df_mg = gdf.merge(df_avf, on='roi')

# %%
f, ax_arr = plot_surface(df_mg,
             column='Offset',
             cmap='Reds',
             show_cbar=True)

f.savefig(out_p / 'offset_bp.svg')


#%%
f, ax_arr = plot_surface(df_mg,
             column='Exponent_1_log',
             cmap='Reds',
             show_cbar=True)
f.savefig(out_p / 'exp1_bp.svg')


# %%
f, ax_arr = plot_surface(df_mg,
             column='Exponent_2',
             cmap='Reds',
             show_cbar=True)

f.savefig(out_p / 'exp2_bp.svg')
# %%
f, ax_arr = plot_surface(df_mg,
             column='tau_log',
             cmap='Reds',
             show_cbar=True)
f.savefig(out_p / 'tau_bp.svg')

# %% plot periodics 
train_peaks = np.logical_and(np.isclose(df['beta_cf'], 16.666, atol=.4), 
                             np.isclose(df['beta_bw'], 1, atol=.5))

df_p = df.copy()

df_p['beta_cf'][train_peaks] = np.nan
df_p['beta_pw'][train_peaks] = np.nan
df_p['beta_bw'][train_peaks] = np.nan
df_p['n_peaks'][train_peaks] = df['n_peaks'] - 1


df_p.rename({'ch_name': 'roi'}, axis=1, inplace=True)

#%%
f, ax = plt.subplots(figsize=(5,5))
sns.histplot(data=df_p, x='n_peaks', hue='tinnitus', 
             hue_order=[False, True],
             ax=ax, bins=8)
sns.despine()

f.savefig(out_p / 'n_peaks_hist.svg')
#%%
df_nt = df_p.query('tinnitus == False')
df_t = df_p.query('tinnitus == True')

nt_cfs = np.concatenate(df_nt[['delta_cf', 'theta_cf', 'alpha_cf', 'beta_cf']].values)
t_cfs = np.concatenate(df_t[['delta_cf', 'theta_cf', 'alpha_cf', 'beta_cf']].values)

#%%
bins = 90
f, ax = plt.subplots(figsize=(10,5))
ax.hist(t_cfs, bins=bins, alpha=0.5, color=tin_c, label='tinnitus')
ax.hist(nt_cfs, bins=bins, alpha=0.5, color=no_tin_c, label='control')

plt.legend()
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('# Oscillations')
ax.set_xlim(1, 30)
sns.despine()
f.savefig(out_p / 'oscillations_freqs.svg')



#%%
df_avf = df_p.groupby('roi').mean(numeric_only=True).reset_index()
gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
              atlas_name= 'glasser')
df_mg = gdf.merge(df_avf, on='roi')

f, _ = plot_surface(df_mg,
             column='n_peaks',
             cmap='Reds',
             show_cbar=True)
f.savefig(out_p / 'n_peaks_bp.svg')

print(f'max n peaks: {df_mg['n_peaks'].max()}')
print(f'min n peaks: {df_mg['n_peaks'].min()}')

# %% plot oscillation scores

df_avf = df_p.query('tinnitus == False').groupby('roi').mean(numeric_only=True).reset_index()
gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
              atlas_name= 'glasser')
df_mg = gdf.merge(df_avf, on='roi')

#%%
df_mg['osc_score_delta'] = df_mg['delta_osc'] * (df_mg['delta_pw'] / np.max(df_mg['delta_pw']))
df_mg['osc_score_theta'] = df_mg['theta_osc'] * (df_mg['theta_pw'] / np.max(df_mg['theta_pw']))
df_mg['osc_score_alpha'] = df_mg['alpha_osc'] * (df_mg['alpha_pw'] / np.max(df_mg['alpha_pw']))
df_mg['osc_score_beta'] = df_mg['beta_osc'] * (df_mg['beta_pw'] / np.max(df_mg['beta_pw']))

#%% probability of alpha

f, _ = plot_surface(df_mg,
             column='alpha_osc',
             cmap='Reds',
             vmin=0.6,
             vmax=1,
             show_cbar=True)
f.savefig(out_p / 'p_alpha_no_tinn.svg')

# %%

vmin, vmax = 0, 0.9

f, _ = plot_surface(df_mg,
             column='osc_score_delta',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)

f.savefig(out_p / 'osc_score_delta_no_tinn.svg')
# %%
f, _ = plot_surface(df_mg,
             column='osc_score_theta',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)

f.savefig(out_p / 'osc_score_theta_no_tinn.svg')
# %%
f, _ = plot_surface(df_mg,
             column='osc_score_alpha',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)
f.savefig(out_p / 'osc_score_alpha_no_tinn.svg')
# %%
f, _ = plot_surface(df_mg,
             column='osc_score_beta',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)

f.savefig(out_p / 'osc_score_beta_no_tinn.svg')
# %%
# %% plot oscillation scores tinnitus

df_avf = df_p.query('tinnitus == True').groupby('roi').mean(numeric_only=True).reset_index()
gdf = rda2gpd(path2atlas=None,#'../data/glasser.rda', 
              atlas_name= 'glasser')
df_mg = gdf.merge(df_avf, on='roi')

#%%
df_mg['osc_score_delta'] = df_mg['delta_osc'] * (df_mg['delta_pw'] / np.max(df_mg['delta_pw']))
df_mg['osc_score_theta'] = df_mg['theta_osc'] * (df_mg['theta_pw'] / np.max(df_mg['theta_pw']))
df_mg['osc_score_alpha'] = df_mg['alpha_osc'] * (df_mg['alpha_pw'] / np.max(df_mg['alpha_pw']))
df_mg['osc_score_beta'] = df_mg['beta_osc'] * (df_mg['beta_pw'] / np.max(df_mg['beta_pw']))

vmin, vmax = 0, 0.9


#%%
f, _ = plot_surface(df_mg,
             column='alpha_osc',
             cmap='Reds',
             vmin=0.6,
             vmax=1,
             show_cbar=True)
f.savefig(out_p / 'p_alpha_tinn.svg')


#%%
f, _ = plot_surface(df_mg,
             column='osc_score_delta',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)

f.savefig(out_p / 'suppl_osc_score_delta_tinn.svg')
# %%
f, _ = plot_surface(df_mg,
             column='osc_score_theta',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)

f.savefig(out_p / 'suppl_osc_score_theta_tinn.svg')
# %%
f, _ = plot_surface(df_mg,
             column='osc_score_alpha',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)
f.savefig(out_p / 'suppl_osc_score_alpha_tinn.svg')
# %%
f, _ = plot_surface(df_mg,
             column='osc_score_beta',
             cmap='Reds',
             vmax=vmax,
             vmin=vmin,
             show_cbar=True)

f.savefig(out_p / 'suppl_osc_score_beta_tinn.svg')
# %%
