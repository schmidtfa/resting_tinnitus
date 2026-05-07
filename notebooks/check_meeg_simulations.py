#%%
import pandas as pd
from pathlib import Path

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(context='poster',
              style='ticks',
              palette='deep',
              )

from ggseg_py.ggseg_py import rda2gpd, merge_data
from ggseg_py.plotting_utils import plot_surface, plot_view

# %%
bp = Path('/home/schmidtfa/experiments/resting_tinnitus/data/sim_pool/slope_varying')

all_files = list(bp.glob('sim*.csv'))
# %%

dfs = []
for f in all_files:

    full_f_name = str(f)

    area = '_'.join(full_f_name.split('/')[-1].split('_')[1:-1])
    cur_df = pd.read_csv(full_f_name, index_col=0)
    cur_df['area'] = area

    dfs.append(cur_df)
# %%
sim_df = pd.concat(dfs)
# %%
sim_corr = pd.DataFrame({
            'Unpooled-Model' : (sim_df['lm_unpooled_correctly_detected'] / sim_df['effs_simulated']) * 100,
              'LME-Model' : (sim_df['lme_classic_correctly_detected'] / sim_df['effs_simulated']) * 100,
              'Cortex-LME-Model' : (sim_df['lme_correctly_detected'] / sim_df['effs_simulated']) * 100,
             }).melt()

#%%
f, ax = plt.subplots(figsize=(10,5))
#sns.violinplot(sim_corr, y='variable', x='value', hue='variable')
sns.boxenplot(sim_corr, y='variable', x='value', hue='variable', ax=ax)

ax.set_ylabel('')
ax.set_xlabel('Effects Correctly Identified (%)')

sns.despine()

base_path = '/home/schmidtfa/experiments/resting_tinnitus/results/final_simulation/'

f.savefig(base_path + 'corr_effs_sim.svg')
# %%
sim_df['LME-Model'] = sim_df['lme_classic_incorrectly_detected'] / (360 - sim_df['effs_simulated']) * 100
sim_df['Cortex-LME-Model'] = sim_df['lme_incorrectly_detected'] / (360 - sim_df['effs_simulated']) * 100
sim_df['Unpooled-Model'] = sim_df['lm_unpooled_incorrectly_detected'] / (360 - sim_df['effs_simulated']) * 100
# %%
sim_incorr = sim_df[['Unpooled-Model',
                     'LME-Model', 
                     'Cortex-LME-Model', 
                     
                     ]].melt()
# %%
f, ax = plt.subplots(figsize=(10,5))
sns.boxenplot(sim_incorr, y='variable', x='value', hue='variable', ax=ax)
#sns.displot(sim_incorr, y='variable', x='value', hue='variable', ax=ax)
ax.set_ylabel('')
ax.set_xlabel('Effects Incorrectly Identified (%)')

sns.despine()

f.savefig(base_path + 'incorr_effs_sim.svg')

# %% here we look at the effects
all_files = list(bp.glob('eff*.csv'))
# %%

dfs = []
for f in all_files:

    full_f_name = str(f)

    area = '_'.join(full_f_name.split('/')[-1].split('_')[1:-1])
    cur_df = pd.read_csv(full_f_name, index_col=0)
    cur_df['area'] = area

    dfs.append(cur_df)
# %%

df_cmb_eff = pd.concat(dfs)



#%%
df_cmb_eff['sim_slopes']


#%%
L = df_cmb_eff['mean'] - 2 * df_cmb_eff['sd']
U = df_cmb_eff['mean'] + 2 * df_cmb_eff['sd']

df_cmb_eff['covered'] = (L <= df_cmb_eff['sim_slopes']) & (df_cmb_eff['sim_slopes'] <= U)
df_cmb_eff['miss_low']      = df_cmb_eff['sim_slopes'] < L     # overestimation
df_cmb_eff['miss_high ']    = df_cmb_eff['sim_slopes'] > U 

df_cmb_eff_true = df_cmb_eff.query('sim_slopes > 0')
df_cmb_eff_false = df_cmb_eff.query('sim_slopes < 0.001')

true_cov = df_cmb_eff_true.groupby(['model', 'roi']).mean(numeric_only=True).reset_index()
false_cov = df_cmb_eff_false.groupby(['model', 'roi']).mean(numeric_only=True).reset_index()

df4plot = true_cov[['model', 'roi', 'covered']].merge(false_cov[['model', 'roi', 'covered']], on=['model', 'roi'])

df4plot['total'] = (df4plot['covered_x'] + df4plot['covered_y']) / 2

#%%

gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df4plot.query('model == "unpooled"'), on='roi')

f, ax = plot_surface(df_mg,
             column='total',
             cmap='Reds', 
             show_cbar=True,
             vmin=0.89,
             vmax=1,
             )

f.savefig(base_path + 'positive_coverage_lm.svg')

#%%
gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df4plot.query('model == "lme_classic"'), on='roi')

f, ax = plot_surface(df_mg,
             column='total',
             cmap='Reds', 
             show_cbar=True,
             vmin=0.89,
             vmax=1,
             )

f.savefig(base_path + 'positive_coverage_lme_classy.svg')

#%%
gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df4plot.query('model == "lme_cortical"'), on='roi')

f, ax = plot_surface(df_mg,
             column='total',
             cmap='Reds', 
             show_cbar=True,
             vmin=.89,
             vmax=1,
             )

f.savefig(base_path + 'positive_coverage_lme_cortical.svg')


#%% now


true_cov_melt = (true_cov.query('covered < .89')[['miss_low', 'miss_high ', 'model']]
                         .melt(id_vars='model'))


true_cov_melt['model'] = true_cov_melt['model'].replace({'lm': 'Unpooled-Model',
                                'lme_classic': 'LME-Model', 
                                'lme_cortical': 'Cortex-LME-Model',})

true_cov_melt['variable'] = true_cov_melt['variable'].replace({'miss_low': 'underestimation',
                                                               'miss_high ': 'overestimation'})

g = sns.FacetGrid(data=true_cov_melt, col='model', hue='variable', height=6, sharey=False)
g.map_dataframe(sns.violinplot, x='variable', y='value')

for ax in g.axes[0]:
    ax.set_xlabel('')
    

g.axes[0][0].set_ylabel('miscalibrated effects \n (Coverage < 0.89; %)')



g.savefig(base_path + 'over_under_covered.svg')



#%%
df4plot = df_cmb_eff_false.groupby(['model', 'roi']).mean(numeric_only=True).reset_index()


gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df4plot.query('model == "unpooled"'), on='roi')

f, ax = plot_surface(df_mg,
             column='covered',
             cmap='Reds', 
             show_cbar=True,
             vmin=0.89,
             vmax=1,
             )

f.savefig(base_path + 'neg_coverage_lm.svg')

#%%
gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df4plot.query('model == "lme_classic"'), on='roi')

f, ax = plot_surface(df_mg,
             column='covered',
             cmap='Reds', 
             show_cbar=True,
             vmin=0.89,
             vmax=1,
             )

f.savefig(base_path + 'neg_coverage_lme_classy.svg')

#%%
gdf = rda2gpd(path2atlas='../data/glasser.rda', atlas_name= 'glasser')
df_mg = gdf.merge(df4plot.query('model == "lme_cortical"'), on='roi')

f, ax = plot_surface(df_mg,
             column='covered',
             cmap='Reds', 
             show_cbar=True,
             vmin=0.89,
             vmax=1,
             )

f.savefig(base_path + 'neg_coverage_lme_cortical.svg')

# %%
df_cmb_eff.groupby('model')[['mean_mask', 'sim_slopes']].corr()
# %%
df_cmb_eff[df_cmb_eff['sim_slopes'] != 0].groupby('model')[['mean_mask', 'sim_slopes']].corr()

# %% visualize probability of correctly identifiying the simulated effect size
detected_effects = df_cmb_eff[df_cmb_eff['sim_slopes'] != 0]

detected_effects['effect_size_correct'] = np.isclose(detected_effects['mean'], 
                                                     detected_effects['sim_slopes'], 
                                                     atol=detected_effects['sd']*2) * 100

f, ax = plt.subplots(figsize=(5,5))
sns.barplot(detected_effects, 
              y='model', 
              x='effect_size_correct', 
              hue='model',
              ax=ax)

ax.set_yticklabels(['unpooled', 
                    'partial pooling \n (cortical areas)', 
                    'partial pooling \n (whole brain)'])
ax.set_ylabel('')
ax.set_xlabel('Simulated Effect Size \n estimated within 2SD of mean (%)')
sns.despine()

#%% check if incorrectly effect sizes are over estimated
incorrect_effs = detected_effects[detected_effects['effect_size_correct'] == 0]

incorrect_effs['overestimation'] = ((incorrect_effs['mean'] - incorrect_effs['sd']*2) > incorrect_effs['sim_slopes']) * 100
incorrect_effs['underestimation'] = ((incorrect_effs['mean'] - incorrect_effs['sd']*2) < incorrect_effs['sim_slopes']) * 100


f, ax = plt.subplots(figsize=(5,5))
sns.barplot(incorrect_effs, 
              y='model', 
              x='overestimation', 
              hue='model',
              ax=ax)

ax.set_yticklabels(['unpooled', 'partial pooling \n (cortical areas)', 'partial pooling \n (whole brain)'])
ax.set_ylabel('')
ax.set_xlabel('Simulated Effect Size \n overestimated 2SD of mean (%)')
sns.despine()


#%%

f, ax = plt.subplots(figsize=(5,5))
sns.barplot(incorrect_effs, 
              y='model', 
              x='underestimation', 
              hue='model',
              ax=ax)

ax.set_yticklabels(['unpooled', 'partial pooling \n (cortical areas)', 'partial pooling \n (whole brain)'])
ax.set_ylabel('')
ax.set_xlabel('Simulated Effect Size \n underestimated 2SD of mean (%)')
sns.despine()

# %%
effs_by_roi = detected_effects.groupby(['model', 'roi'])['effect_size_correct'].mean()
effs_by_roi.reset_index().query('model == "lme_cortical"')
# %%
effs_by_roi.reset_index().query('model == "lme_classic"')
# %%

