#%%

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

import pandas as pd
import joblib

import seaborn as sns
sns.set_style('ticks')
sns.set_context('poster')
from os.path import join

import matplotlib.pyplot as plt

#%%
data = joblib.load('/mnt/obob/staff/fschmidt/resting_tinnitus/data/rsa_irasa_perm_test.dat')

#%%
rdm_p_f = data['frequency']['periodic']['real']
rdm_ap_f = data['frequency']['aperiodic']['real']

rand_rdm_f_p = data['frequency']['periodic']['random']
rand_rdm_f_ap = data['frequency']['aperiodic']['random']

#%%

f, axes = plt.subplots(figsize=(16, 8), ncols=2)

#sns.lineplot(df_rsa_raw, x='Frequency', 
 #            y='Dissimilarity', hue='tinnitus', ax=axes[0])
sns.lineplot(data=rdm_p_f.query('Frequency < 25.25'), 
             x='Frequency', y='Dissimilarity',errorbar=('pi', 95), #hue='tinnitus', 
             ax=axes[0], legend=False)
sns.lineplot(data=rand_rdm_f_p.query('Frequency < 25.25'), 
             x='Frequency', y='Dissimilarity', errorbar=('pi', 95),  #hue='tinnitus', 
             ax=axes[0], legend=False)

sns.lineplot(data=rdm_ap_f, x='Frequency', 
             y='Dissimilarity', errorbar=('pi', 95),# hue='tinnitus', 
             ax=axes[1], legend=False)

sns.lineplot(data=rand_rdm_f_ap, x='Frequency', 
             y='Dissimilarity', errorbar=('pi', 95), # hue='tinnitus', 
             ax=axes[1], legend=False)

#axes[0].set_title('Raw Spectrum')
axes[0].set_title('Periodic Spectrum')
#axes[0].set_xlim(15, 30)
axes[1].set_title('Aperiodic Spectrum')

#axes[0].set_yscale('log')
#axes[0].set_xscale('log')
#axes[1].set_ylabel('')
#axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[1].set_ylabel('')
plt.tight_layout()
sns.despine()
#f.savefig('../results/dissimilarity_metric_euclidean.svg')
#%%
#%%
rdm_p_s = data['spatial']['periodic']['real']
rdm_ap_s = data['spatial']['aperiodic']['real']

rand_rdm_s_p = data['spatial']['periodic']['random']
rand_rdm_s_ap = data['spatial']['aperiodic']['random']



#%%

cmap = sns.color_palette('deep')

#%%
sns.set_context('talk')
f, ax = plt.subplots(figsize=(5, 100))
sns.pointplot(data=rdm_p_s, 
              y='ch_name', 
              x='Dissimilarity', 
              markers='|',
              color=cmap[0],
              #errorbar='se',
              #dodge=True,
              ax=ax, 
              join=False,
              )

sns.pointplot(data=rand_rdm_s_p, 
              y='ch_name', 
              x='Dissimilarity', 
              color=cmap[1],
              markers='|',
              errorbar=('pi', 95), 
              ax=ax,
              join=False)

#ax.set_xscale('log')

#%%
sns.set_context('talk')
f, ax = plt.subplots(figsize=(5, 100))
sns.pointplot(data=rdm_ap_s, 
              y='ch_name', 
              x='Dissimilarity', 
              markers='|',
              color=cmap[0],
              #errorbar='se',
              #dodge=True,
              ax=ax, 
              join=False,
              )

sns.pointplot(data=rand_rdm_s_ap, 
              y='ch_name', 
              x='Dissimilarity', 
              color=cmap[1],
              markers='|',
              errorbar=('pi', 95), 
              ax=ax,
              join=False)


#%%
rdm_ap_s.to_csv('../results/dissimilarity_aperiodic_spatial.csv')
rdm_p_s.to_csv('../results/dissimilarity_periodic_spatial.csv')
