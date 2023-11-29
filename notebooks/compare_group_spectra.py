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

local = True

no_knee = True #knee modeling seems to make not too much sense given there are mostly no knees after beamforming

if local:
    home_base = '/Users/b1059770/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes.noindex/bomber/resting_tinnitus'
else:
    home_base = '/mnt/obob/staff/fschmidt/resting_tinnitus/'

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
sys.path.append(join(home_base, 'utils'))

from src_utils import plot_parc
from specparam_utils import knee_or_fixed

sns.set_style('ticks')
sns.set_context('poster')

cmap = sns.color_palette('deep').as_hex()

tin_c = cmap[1]
no_tin_c = cmap[0]


#%%
all_files = list(Path(join(home_base, 'data/specparam_3')).glob(f'*/*__peak_threshold_2__freq_range_[[]0.25, 98[]].dat'))

# %%
periodic, aperiodic = [], []

for f in all_files:
    
    try: 
        cur_data = joblib.load(f)

        periodic.append(cur_data['periodic'])
        aperiodic.append(cur_data['aperiodic'])

        freq = cur_data['freq']
        label_info = cur_data['label_info']
    except:
        print(str(f).split('/')[-1])
# %%
physio_chs = ['ECG', 'EOGH', 'EOGV']

df_p = pd.concat(periodic).query('ch_name != @physio_chs')
df_ap = pd.concat(aperiodic).query('ch_name != @physio_chs')

#%%
df_ap_knee = knee_or_fixed(df_ap.copy())

#%% plot aperiodic activity on parcellation
df_ap_g = df_ap_knee.groupby('ch_name').mean()
#%%
cur_param = "knee_modeled"
tinnitus = False

df2plot = df_ap_knee.query(f'tinnitus == {tinnitus}').groupby('ch_name').mean().reset_index()

plot_kwargs = {
    'hemi':"split",
    'surf':"inflated",
    'views':["lateral", "medial"],
    'subjects_dir':subjects_dir,
    'cortex':[(.6,.6,.6), (.6,.6,.6)], #turn sulci and gyri to the same grey
    'background':'white',
    #'show_toolbar':False, 
    'offscreen':True,
    'size':(800, 400),
}

stc_parc = np.concatenate([df2plot.query('ch_name == "???"')[cur_param], df2plot[cur_param]])
stc_mask = np.zeros(stc_parc.shape[0]) == 1
stc_mask[:2] = True #mask subcortical


cur_eff = plot_parc(stc_parc, 
          stc_mask, 
          labels_mne, 
          subjects_dir, 
          cmap='magma', 
          clevels=(df2plot[cur_param].min(),
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

fig.savefig(f'../results/brain_{cur_param}_tinnitus_{tinnitus}.svg')

#%%
df2plot[cur_param].max()
#%%
df2plot[cur_param].min()

#%% select knee channels

if no_knee:

    df_ap_new = df_ap.query('aperiodic_mode == "fixed"')

else:

    knee_settings = joblib.load(join(home_base, 'data/knee_settings.dat'))
    knee_chans = knee_settings['knee']
    fixed_chans = knee_settings['fixed']

    df_ap_new = pd.concat([df_ap.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                        df_ap.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])

#%% plot some histograms
#start with tinnitus distress distribution
tinn_distress = df_ap_new.drop_duplicates('subject_id').query('tinnitus == True')[['subject_id', 'tinnitus_distress']]
%matplotlib inline

fig, ax = plt.subplots(figsize=(4,4))
ax.hist(tinn_distress['tinnitus_distress'], bins=15, density=False, color=tin_c,)
ax.set_ylabel('Count')
ax.set_xlabel('Tinnitus Distress')
sns.despine()
fig.savefig(f'../results/tinnitus_distress_hist.svg')

#%% 


fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_ap_new.query('tinnitus == False')['r_squared'], color=no_tin_c, edgecolor=no_tin_c, bins=1000)
ax.set_ylabel('Count')
ax.set_xlabel('R2')
ax.set_xlim(.60, 1)
plt.axvline(0.9, color='#CC5555', label='cut-off', linestyle='--')
plt.legend()
#ax.set_ylim(0, 10000)
sns.despine()
fig.savefig(f'../results/r_squared_hist_no_tinnitus.svg')


#%% exponent
#mask bad channels
df_ap_new = df_ap_new.mask(df_ap_new['r_squared'] < .90)
fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_ap_new.query('tinnitus == False')['exponent'], color=no_tin_c, edgecolor=no_tin_c, bins=50)
ax.set_ylabel('Count')
ax.set_xlabel('Exponent')
#ax.set_xlim(0, 5)
#ax.set_ylim(0, 3000)

sns.despine()
fig.savefig(f'../results/exponent_hist_no_tinnitus.svg')


#%% knee freq

if no_knee == False:

    knee_settings = joblib.load(join(home_base, 'data/knee_settings.dat'))
    knee_chans = knee_settings['knee']
    fixed_chans = knee_settings['fixed']

    df_ap_2 = pd.concat([df_ap.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                        df_ap.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])

    fig, ax = plt.subplots(figsize=(4,4))

    ax.hist(df_ap_2.query('tinnitus == False')['knee_freq'], color=no_tin_c, edgecolor=no_tin_c, bins=20)
    ax.set_ylabel('Count')
    ax.set_xlabel('Knee Frequency (Hz)')
    ax.set_xlim(0, 50)
    #ax.set_ylim(0, 100)
    #ax.set_yscale('log')
    sns.despine()
    fig.savefig(f'../results/knee_freq_hist_no_tinnitus.svg')


#%% offset

fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_ap_new.query('tinnitus == False')['offset'], color=no_tin_c, edgecolor=no_tin_c, bins=100)
ax.set_ylabel('Count')
ax.set_xlabel('Power')
#ax.set_xlim(-30, -15)
#ax.set_ylim(0, 10000)
sns.despine()
fig.savefig(f'../results/offset_hist_no_tinnitus.svg')


#%% plot aperiodic activity on parcellation
df_ap_g = df_ap_new.groupby('ch_name').mean()
#%%
cur_param = "offset"
tinnitus = False

df2plot = df_ap_new.query(f'tinnitus == {tinnitus}').groupby('ch_name').mean().reset_index()

plot_kwargs = {
    'hemi':"split",
    'surf':"inflated",
    'views':["lateral", "medial"],
    'subjects_dir':subjects_dir,
    'cortex':[(.6,.6,.6), (.6,.6,.6)], #turn sulci and gyri to the same grey
    'background':'white',
    #'show_toolbar':False, 
    'offscreen':True,
    'size':(800, 400),
}

stc_parc = np.concatenate([df2plot.query('ch_name == "???"')[cur_param], df2plot[cur_param]])
stc_mask = np.zeros(stc_parc.shape[0]) == 1
stc_mask[:2] = True #mask subcortical


cur_eff = plot_parc(stc_parc, 
          stc_mask, 
          labels_mne, 
          subjects_dir, 
          cmap='magma', 
          clevels=(df2plot[cur_param].min(),
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

fig.savefig(f'../results/brain_{cur_param}_tinnitus_{tinnitus}.svg')

#%%
# %%
df2plot[cur_param].min()
# %%
df2plot[cur_param].max()

# %% now lets go periodic


#%% histograms of center frequencies
%matplotlib inline
df_cf = df_p.query('peak_params == "cf"')


if no_knee:

    df_cf_new = df_cf.query('aperiodic_mode == "fixed"')

else:


    df_cf_new = pd.concat([df_cf.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                    df_cf.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])

#remove line frequency peak from oscillations
df_cf_new['n_peaks'] = df_cf_new['n_peaks'] - (np.isnan(df_cf_new['line_noise']) == False).to_numpy().astype(int)
df_cf_new['n_peaks'] = df_cf_new['n_peaks'] - np.logical_and(df_cf_new['beta'] < 17, df_cf_new['beta'] > 16).to_numpy().astype(int)

df_cf_new['beta'][np.logical_and(df_cf_new['beta'] < 17, df_cf_new['beta'] > 16).to_numpy()] = np.nan

df_cf_new = df_cf_new.mask(df_cf_new['r_squared'] < .90)

fig, ax = plt.subplots(figsize=(4,4))

ax.hist(df_cf_new.query('tinnitus == False')['n_peaks'], color=no_tin_c, edgecolor=no_tin_c, bins=10)
ax.set_ylabel('Count')
ax.set_xlabel('# Peaks')
ax.set_xlim(0, 6)
#ax.set_ylim(0, 8000)
sns.despine()
fig.savefig(f'../results/n_peaks_hist_no_tinnitus.svg')


#%%
cf_tin = np.concatenate(df_cf_new.query('tinnitus == True')[['delta', 'theta', 'alpha', 'beta', 'gamma']].to_numpy())
cf_no_tin = np.concatenate(df_cf_new.query('tinnitus == False')[['delta', 'theta', 'alpha', 'beta', 'gamma']].to_numpy())

#%%
fig, ax = plt.subplots(figsize=(8,4))

ax.hist(cf_no_tin, color=no_tin_c, edgecolor=no_tin_c, bins=30)
ax.set_ylabel('Count')
ax.set_xlabel('Frequency (Hz)')
ax.set_xlim(0, 45)
#ax.set_ylim(0, 4000)
sns.despine()
fig.savefig(f'../results/cf_hist_no_tinnitus.svg')


# %% plot periodic power

df_pw = df_p.query('peak_params == "pw"')

if no_knee:

    df_pw_new = df_pw.query('aperiodic_mode == "fixed"')

else:
    df_pw_new = pd.concat([df_pw.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                    df_pw.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])
df_cf = df_p.query('peak_params == "cf"')

if no_knee:

    df_cf_new = df_cf.query('aperiodic_mode == "fixed"')

else:
    df_cf_new = pd.concat([df_cf.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                    df_cf.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])

df_pw_new.reset_index()[['delta', 'theta', 'alpha', 'beta', 'gamma']]

df_pw_new['n_peaks'] = df_pw_new['n_peaks'] - (np.isnan(df_pw_new['line_noise']) == False).to_numpy().astype(int)
df_pw_new['n_peaks'] = df_pw_new['n_peaks'] - np.logical_and(df_cf_new['beta'] < 17, df_cf_new['beta'] > 16).to_numpy().astype(int)
df_pw_new['beta'][np.logical_and(df_cf_new['beta'] < 17, df_cf_new['beta'] > 16).to_numpy()] = np.nan
df_pw_new = df_pw_new.mask(df_pw_new['r_squared'] < .90)

df_pw_new['n_oscillations'] = (np.isnan(df_pw_new.reset_index()[['delta', 'theta', 'alpha', 'beta', 'gamma']]) == False).sum(axis=1)

#lambda x, feature: np.isnan(x[feature]) == False
for feature in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    df_pw_new[f'peaks_{feature}'] = np.isnan(df_pw_new[feature]) == False

    df_pw_new[f'{feature}_score'] = df_pw_new[f'peaks_{feature}'] * df_pw_new[f'{feature}']

#%%
df_pw_new

#%%
cur_param = "n_peaks"
tinnitus = False


df2plot = df_pw_new.query(f'tinnitus == {tinnitus}').groupby('ch_name').mean().reset_index()

plot_kwargs = {
    'hemi':"split",
    'surf':"inflated",
    'views':["lateral", "medial"],
    'subjects_dir':subjects_dir,
    'cortex':[(.6,.6,.6), (.6,.6,.6)], #turn sulci and gyri to the same grey
    'background':'white',
    #'show_toolbar':False, 
    'offscreen':True,
    'size':(800, 400),
}

stc_parc = np.concatenate([df2plot.query('ch_name == "???"')[cur_param], df2plot[cur_param]])
stc_mask = np.zeros(stc_parc.shape[0]) == 1
stc_mask[:2] = True #mask subcortical


if cur_param == "n_peaks":
    clevels = (df2plot[cur_param].min(),
               df2plot[cur_param].mean(),
               df2plot[cur_param].max())
else:
    clevels=(0,
             0.5,
             1)

cur_eff = plot_parc(stc_parc, 
          stc_mask, 
          labels_mne, 
          subjects_dir, 
          cmap='magma',
          clevels=clevels,
          plot_kwargs=plot_kwargs, 
          parc='HCPMMP1')



fig, ax = plt.subplots()
ax.axis("off")

plt.imshow(cur_eff, cmap='magma')
cbaxes = inset_axes(plt.gca(), width="6%", height="36%", loc=7, borderpad=-2)
cbar = plt.colorbar(cax=cbaxes, ax=ax, orientation='vertical')
#plt.clim(0, 1)
#plt.tight_layout()
plt.show()
fig.tight_layout()

fig.savefig(f'../results/brain_{cur_param}_tinnitus_{tinnitus}.svg')
# %%
df2plot[cur_param].min()
# %%
df2plot[cur_param].max()
# %%
