#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


df_matched = pd.read_csv('../data/tinnitus_match.csv')
#%%
fig, ax = plt.subplots(figsize=(3, 6))

sns.swarmplot(data=df_matched, x='tinnitus', y='measurement_age', palette='deep',
              hue='tinnitus', ax=ax, size=10, alpha=0.4, )#legend=False)
sns.pointplot(data=df_matched, x='tinnitus', y='measurement_age', 
              hue='tinnitus', palette='deep',
              ax=ax, estimator='mean', markers='_', linestyle="none",
              markersize=30,  markeredgewidth=5,)

ax.set_ylabel('age (years)')
sns.despine()
fig.savefig('../results/age_diff_tin_con.svg')

# %%
fig, ax = plt.subplots(figsize=(3, 6))
sns.swarmplot(data=df_matched, x='tinnitus', y='dB', palette='deep',
              hue='tinnitus', ax=ax, size=10, alpha=0.4,)# legend=False)
sns.pointplot(data=df_matched, x='tinnitus', y='dB', 
              hue='tinnitus', palette='deep',
              ax=ax, estimator='mean', markers='_', linestyle="none",
              markersize=30,  markeredgewidth=5,)
ax.set_ylabel('Hearing Threshold (dB)')
sns.despine()

fig.savefig('../results/pta_match.svg')
# %%
fig, ax = plt.subplots(figsize=(3, 6))
cross_tab_pct = pd.crosstab(index=df_matched['tinnitus'], columns=df_matched['gender'], normalize='index')
cross_tab_pct.plot(kind='bar', stacked=True, ax=ax, color=[sns.color_palette('deep')[3],
                                                            sns.color_palette('deep')[4]])
ax.set_xticklabels(labels=[False, True],rotation=0)
ax.set_ylabel('Proportion')
sns.despine()
fig.savefig('../results/gender_match.svg')
# %%
