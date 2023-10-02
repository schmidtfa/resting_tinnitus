#%%
from pathlib import Path
from os.path import join
import joblib
import pandas as pd
import mne
import numpy as np
from scipy.stats import zscore
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns
import sys

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set_context('poster')

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

local = False

if local:
    home_base = '/Users/b1059770/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes.noindex/bomber/resting_tinnitus'
else:
    home_base = '/mnt/obob/staff/fschmidt/resting_tinnitus/'


#% get source space stuff

data_dir = join(home_base, 'data/log_reg/')

trans_path = 'data/headmodels/'
mri_path = 'data/freesurfer/'
fs_path = join(home_base, mri_path, 'fsaverage')


src_file = join(fs_path, 'bem', 'fsaverage-ico-4-src.fif')
subjects_dir = join(home_base, 'data/freesurfer/')

src = mne.read_source_spaces(src_file)
labels_mne = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

names_order_mne = np.array([label.name[:-3] for label in labels_mne])

# add plotting functions
sys.path.append(join(home_base, 'utils'))

from src_utils import plot_parc
from specparam_utils import knee_or_fixed

sns.set_style('ticks')
sns.set_context('poster')

cmap = sns.color_palette('deep').as_hex()

tin_c = cmap[1]
no_tin_c = cmap[0]

#define bad subjects (aka source reconstruction failed)
bad_subjects = ['19541130anfn', '19590423mrbr', 
                '19761120eitn', '19910703eigl',
                '19930120laat', '19930506urhe',
                '19930709crgl', '19930727agwl',
                ]

#%%
all_files = list(Path(join(home_base, 'data/specparam_train')).glob(f'*/*_freq_range_[[]0.5, 100[]].dat'))


# %%
periodic, aperiodic = [], []

for f in all_files:
    
    cur_data = joblib.load(f)

    periodic.append(cur_data['periodic'])
    aperiodic.append(cur_data['aperiodic'])

    freq = cur_data['freq']
    label_info = cur_data['label_info']
# %%
physio_chs = ['ECG', 'EOGH', 'EOGV']

df_p = pd.concat(periodic).query('subject_id != @bad_subjects').query('ch_name != @physio_chs')
df_ap = pd.concat(aperiodic).query('subject_id != @bad_subjects').query('ch_name != @physio_chs')


#%% get bic for knee
order_mne = cur_data['label_info']['names_order_mne']

#%%
df_ap_new = knee_or_fixed(df_ap)

#%% plot some histograms

#%% exponent
fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_ap_new.query('tinnitus == True')['exponent'], color=tin_c, bins=15)
ax.set_ylabel('Count')
ax.set_xlabel('Exponent')
ax.set_xlim(0, 3)
ax.set_ylim(0, 8000)
sns.despine()
fig.savefig('../results/exponent_hist_tinnitus.svg')

#%%
fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_ap_new.query('tinnitus == False')['exponent'], color=no_tin_c, bins=15)
ax.set_ylabel('Count')
ax.set_xlabel('Exponent')
ax.set_xlim(0, 3)
ax.set_ylim(0, 8000)
sns.despine()
fig.savefig('../results/exponent_hist_no_tinnitus.svg')


#%% knee freq
fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_ap_new.query('tinnitus == True')['knee_freq'], color=tin_c, bins=15)
ax.set_ylabel('Count')
ax.set_xlabel('Frequency (Hz)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 500)
sns.despine()
#fig.savefig('../results/knee_freq_hist_tinnitus.svg')

#%%
fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_ap_new.query('tinnitus == False')['knee_freq'], color=no_tin_c, bins=15)
ax.set_ylabel('Count')
ax.set_xlabel('Frequency (Hz)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 500)
sns.despine()
#fig.savefig('../results/knee_freq_hist_no_tinnitus.svg')

#%% plot aperiodic activity on parcellation
df_ap_g = df_ap_new.groupby('ch_name').mean()
df_ap_no_tin = df_ap_new.query('tinnitus == False').groupby('ch_name').mean()

#%%
cur_param = "exponent"
tinnitus = True

df2plot = df_ap_new.query(f'tinnitus == {tinnitus}').groupby('ch_name').mean().reset_index()

plot_kwargs = {
    'hemi':"split",
    'surf':"inflated",
    'views':["lateral", "medial"],
    'subjects_dir':subjects_dir,
    'cortex':[(.6,.6,.6), (.6,.6,.6)], #turn sulci and gyri to the same grey
    'background':'white',
    'show_toolbar':False, 
    'offscreen':True,
    'size':(800, 400),
}

stc_parc = np.concatenate([df2plot.query('ch_name == "???"')[cur_param], df2plot[cur_param]])
stc_mask = np.zeros(stc_parc.shape[0]) == 1
stc_mask[:2] = True #mask subcortical

#%%

cur_eff = plot_parc(stc_parc, 
          stc_mask, 
          labels_mne, 
          subjects_dir, 
          cmap='magma', 
          clevels=(df_ap_g[cur_param].min(),
                    df_ap_g[cur_param].mean(),
                    df_ap_g[cur_param].max()),
          plot_kwargs=plot_kwargs, 
          parc='HCPMMP1')



fig, ax = plt.subplots()
ax.axis("off")

plt.imshow(eff_brain, cmap='magma')
cbaxes = inset_axes(plt.gca(), width="6%", height="36%", loc=7, borderpad=-2)
cbar = plt.colorbar(cax=cbaxes, ax=ax, orientation='vertical')
#plt.clim(0, 1)
#plt.tight_layout()
plt.show()
fig.tight_layout()

fig.savefig(f'../results/brain_{cur_param}_tinnitus_{tinnitus}.svg')

#%%
df_ap_no_tin


# %%
cur_param = "cf"
tinnitus = True

cur_df = df_p.query(f'peak_params == "{cur_param}"').query(f'tinnitus == {tinnitus}')
# %%
