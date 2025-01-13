#%%
import pandas as pd
import seaborn as sns
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
sns.set_context('poster')
import seaborn.objects as so
from seaborn import plotting_context
from seaborn import axes_style

import os
os.environ["MNE_3D_OPTION_ANTIALIAS"] = "false"

# %%
import mne
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set_context('poster')


import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
from os.path import join
mpl.rcParams.update(new_rc_params)

local = False

if local:
    home_base = '/Users/b1059770/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes.noindex/bomber/resting_tinnitus'
else:
    home_base = '/home/schmidtfa/experiments/resting_tinnitus/'

#% get source space stuff
trans_path = 'data/headmodels/'
mri_path = 'data/freesurfer/'
fs_path = join(home_base, mri_path, 'fsaverage')


src_file = join(fs_path, 'bem', 'fsaverage-ico-4-src.fif')
subjects_dir = join(home_base, 'data/freesurfer/')

src = mne.read_source_spaces(src_file)
labels_mne = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

names_order_mne = np.array([label.name[:-3] for label in labels_mne])

# add plotting functions
import sys
sys.path.append(join(home_base, 'utils'))

from src_utils import plot_parc

#data_dir = join(home_base, 'data/log_reg_duration_6_new/')
data_dir = join(home_base, 'data/log_reg_duration_6_final/')

#%% get data from model
kind = 'up'
feature_dict = {'Offset': None,
                'Exponent_1': None,
                'Exponent_2': None,
                'tau': None,
                #'Knee Frequency (Hz)': None,
                'n_peaks': None,
                'delta': ['pw', 'cf', 'bw'],
                'theta': ['pw', 'cf', 'bw'],
                
                'beta': ['pw', 'cf', 'bw'],
                'alpha': ['pw', 'cf', 'bw'],
                }

all_summaries = []
for key, val in feature_dict.items():

    if val == None:
        mdf = az.from_netcdf(join(data_dir, f'{key}_{kind}.nc'))
        summary = az.summary(mdf, var_names='beta_ch', hdi_prob=.89)
        summary['feature'] = key
        all_summaries.append(summary)
    else:
        for cur_val in val:
            mdf = az.from_netcdf(join(data_dir, f'{key}_{cur_val}_{kind}.nc'))
            summary = az.summary(mdf, var_names='beta_ch', hdi_prob=.89)
            summary['feature'] = key + '_' + cur_val
            all_summaries.append(summary)


# %%
eff_df_cmb = pd.concat(all_summaries)
# %%
rope_lim = 0.185
# 0.185 for rope as suggested by kruschke
eff_df_cmb['positive effect'] = eff_df_cmb['hdi_5.5%'] > rope_lim#0.185
eff_df_cmb['negative effect'] = eff_df_cmb['hdi_94.5%'] < -1*rope_lim#0.185
eff_df_cmb['null effect'] = np.logical_and(eff_df_cmb['hdi_5.5%'] > -1*rope_lim,#-0.185, 
                                           eff_df_cmb['hdi_94.5%'] < rope_lim)#0.185)
#i might need some additional specifications for below
eff_df_cmb['leaning positive effect'] = np.logical_and(eff_df_cmb['mean'] > rope_lim,#0.185, 
                                                       np.logical_and(eff_df_cmb['hdi_5.5%'] > 0., 
                                                                      eff_df_cmb['hdi_5.5%'] < rope_lim))#0.185))
eff_df_cmb['leaning negative effect'] = np.logical_and(eff_df_cmb['mean'] < -1*rope_lim,#-0.185, 
                                                       np.logical_and(eff_df_cmb['hdi_94.5%'] < 0., 
                                                                      eff_df_cmb['hdi_94.5%'] > -1*rope_lim))#-0.185))
#eff_df_cmb['undefined'] = eff_df_cmb[['positive effect', 'negative effect', 'null effect', 'leaning positive effect', 'leaning negative effect']].sum(axis=1) == 0
eff_df_cmb.to_csv(f'../results/hist_log_reg_effects_{kind}.csv')

# %%
#'undefined'
eff_list = (eff_df_cmb[['feature', 'positive effect', 'leaning positive effect', 
                        'negative effect', 'leaning negative effect', 'null effect', 
                        #'undefined'
                        ]].melt(id_vars='feature', 
                                var_name='Effect', 
                                value_name='Observed Effects (%)'))

eff_list['Observed Effects (%)'] *= 100 

eff_list.replace({'feature' : {'delta_cf': 'Delta (cf)', 
                               'delta_pw': 'Delta (pw)', 
                               'delta_bw': 'Delta (bw)', 
                               'theta_cf': 'Theta (cf)', 
                               'theta_pw': 'Theta (pw)', 
                               'theta_bw': 'Theta (bw)', 
                               'alpha_cf': 'Alpha (cf)', 
                               'alpha_pw': 'Alpha (pw)', 
                               'alpha_bw': 'Alpha (bw)', 
                               'beta_cf': 'Beta (cf)', 
                               'beta_pw': 'Beta (pw)',
                               'beta_bw': 'Beta (bw)',
                               'n_peaks': '#Peaks',
                               'Exponent_1': 'Exponent1',
                               'Exponent_2': 'Exponent2',
                               'Offset': 'Offset',
                               }}, inplace=True)
# %%
eff_probas = eff_list.groupby(['Effect', 'feature']).mean().reset_index()#'feature')
#%%


pal = sns.color_palette('deep', as_cmap=True)
pal2 = sns.color_palette('pastel', as_cmap=True)
cmap = [pal[3], pal2[3], pal[0], pal2[0], pal[2]]


c_order = [ 'positive effect', 'leaning positive effect', 'negative effect','leaning negative effect','null effect',]

label_order = ['Delta (cf)', 
            'Delta (pw)', 
            'Delta (bw)', 
           'Theta (cf)', 
           'Theta (pw)', 
           'Theta (bw)', 
           'Alpha (cf)', 
           'Alpha (pw)',
           'Alpha (bw)',
           'Beta (cf)', 
           'Beta (pw)',
           'Beta (bw)',
           '#Peaks', 'Exponent1', 
           'Exponent2', 'tau', 'Offset', 
           ]

# f, ax = plt.subplots(figsize=(8, 4))
# p = (so.Plot(data=eff_probas, 
#             x="feature", 
#             y='Observed Effects (%)', 
#             color="Effect",
#             ymin=0,
#             ymax=100)        
#         #.theme(axes_style("ticks") | plotting_context("talk"))
#         #.layout(size=(8,4))
#         .add(so.Bar(), 
#              so.Stack())
#         .scale(x=so.Nominal(order=label_order),
#                color=so.Nominal(order=c_order, values=cmap),
#                )
#         .label(x="", color="")
#         )#.plot(pyplot=True)


# sns.despine()
# ax.set_xticklabels(label_order, rotation = 90)
# p.on(ax).show()
# #p
# #p.show()
#p.set_xticklabels(p.get_xticks(), rotation = 90)

#p#.show()
#%%

eff_pivot = eff_probas.pivot_table(columns='Effect', index='feature', values='Observed Effects (%)').reset_index()

c_order = ['feature', 'null effect', 'leaning negative effect', 'negative effect', 'leaning positive effect', 'positive effect']

eff_pivot['feature'] = pd.Categorical(eff_pivot['feature'], label_order)
eff_pivot.sort_values('feature', inplace=True)


#%%
#%matplotlib inline
cmap = [pal[2], pal2[0], pal[0],pal2[3], pal[3]]

f, ax = plt.subplots(figsize=(12, 4))
eff_pivot[c_order].plot(
                x='feature',
                kind = 'bar',
                color=cmap,
                stacked = True,
                width=0.8,
                ax=ax)

ax.set_xlabel('')
ax.set_ylabel('Observed Effect (%)')
sns.despine()

f.savefig(f'../results/hist_log_reg_effects_{kind}.eps', format='eps', bbox_inches='tight', transparent=True)

#%% now lets plot everything on a brain
#%matplotlib inline
ch_effects = (eff_df_cmb.reset_index()[['index', 'positive effect', 'negative effect', 'leaning positive effect', 'leaning negative effect']]
           .groupby('index')
           .mean()
           .sum(axis=1)
           .reset_index())

ch_effects.columns = ['ch_name', 'Effect']

ch_effects['ch_name'] = [ch[6:-1] for ch in ch_effects['ch_name']]

#effect_order = [ix[6:-1] for ix in ch_effects.index]

reindex_array = [np.argmax(eff == names_order_mne[2:]) for eff in ch_effects['ch_name']]

#ch_effects[ch_effects['Effect'] <= .14] = np.nan

# %%
cur_param = "Effect"
view = "lateral" # lateral, "medial"
mne.viz.set_3d_backend('notebook')

df2plot = ch_effects#[reindex_array]

plot_kwargs = {
    'hemi':"split",
    'surf':"inflated",
    'views':[view,], # 
    'subjects_dir':subjects_dir,
    'cortex':[(.6,.6,.6), (.6,.6,.6)], #turn sulci and gyri to the same grey
    'background':'white',
    #'show_toolbar':False, 
    #'offscreen':True,
    'size':(800, 400),
    #'block': True,
    'show':False
}

stc_parc = np.concatenate([[0,0], df2plot[cur_param]])
stc_parc[stc_parc == 0] = np.nan
stc_mask = np.zeros(stc_parc.shape[0]) == 1
stc_mask[:2] = True #mask subcortical


cur_eff = plot_parc(stc_parc, 
          stc_mask, 
          labels_mne, 
          subjects_dir, 
          cmap='magma', 
          clevels=(0,
                    df2plot[cur_param].mean(),
                    df2plot[cur_param].max()),
          plot_kwargs=plot_kwargs, 
          parc='HCPMMP1')



fig, ax = plt.subplots()
ax.axis("off")

plt.imshow(cur_eff, cmap='magma')
cbaxes = inset_axes(plt.gca(), width="6%", height="36%", loc=7, borderpad=-2)
cbar = plt.colorbar(cax=cbaxes, ax=ax, orientation='vertical')
plt.clim(df2plot[cur_param].min(), df2plot[cur_param].max())
#plt.tight_layout()
plt.show()
fig.tight_layout()

fig.savefig(f'../results/brain_log_{cur_param}_tinnitus_{view}_almost.svg')

# %%
ch_effects[ch_effects['Effect'] > 0.185]
# %%
ch_effects
# %%
