#%%
from pathlib import Path
import joblib
import pandas as pd

import arviz as az

import mne
import numpy as np
from scipy.stats import zscore
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.special import expit

sns.set_context('poster')

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


local = True
#%%get data
if local:
    home_base = '/Users/b1059770/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes.noindex/bomber/resting_tinnitus'
else:
    home_base = '/mnt/obob/staff/fschmidt/resting_tinnitus'

data_dir = join(home_base, 'data/log_reg/')

trans_path = 'data/headmodels/'
mri_path = 'data/freesurfer/'
fs_path = join(home_base, mri_path, 'fsaverage')


src_file = join(fs_path, 'bem', 'fsaverage-ico-4-src.fif')
subjects_dir = join(home_base, 'data/freesurfer/')

src = mne.read_source_spaces(src_file)
labels_mne = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

names_order_mne = np.array([label.name[:-3] for label in labels_mne])

rh = [True if label.hemi == 'rh' else False for label in labels_mne]
lh = [True if label.hemi == 'lh' else False for label in labels_mne]



#%%
# tinnitus ~ (1 + exponent|channel)
# corr ~ 1 + group + (1 + group|time) + (1|subject)

#%%
#tinnitus ~ (1 + exponent|channel)

#%%

#%% get data from model
feature = 'cf'
freq = 'alpha'

if feature in ['exponent', 'offset', 'knee_freq', 'n_peaks']:
    #ch_effects = pd.read_csv(join(data_dir, f'{feature}.csv'))
    mdf = az.from_netcdf(join(data_dir, f'{feature}.nc'))
else:
    #ch_effects = pd.read_csv(join(data_dir, f'{freq}_{feature}.csv'))
    mdf = az.from_netcdf(join(data_dir, f'{freq}_{feature}.nc'))
ch_effects = az.summary(mdf, var_names='beta|', hdi_prob=.89)

#%%
stc_parc = (np.concatenate([[ch_effects['mean'][0]], ch_effects['mean']])) / 4 #on probability scale
#stc_mask = (np.concatenate([[ch_effects['hdi_5.5%'][0]], ch_effects['hdi_5.5%']])) < 0.185
#stc_mask = (np.concatenate([[ch_effects['hdi_94.5%'][0]], ch_effects['hdi_94.5%']])) > -0.185
stc_mask = np.ones(stc_parc.shape) == 0


#%%

def plot_parc(stc_parc, stc_mask, cmap, parc='HCPMMP1'):

    mpl.use('Qt5Agg')

    plot_kwargs = {
        'hemi':"split",
        'surf':"inflated",
        'views':["lateral", "medial"],
        'subjects_dir':subjects_dir,
        'cortex':[(.6,.6,.6), (.6,.6,.6)], #turn sulci and gyri to the same grey
        'background':'white',
        'offscreen':True,
        'size':(800, 400),
    }


    import nibabel as nib
    Brain = mne.viz.get_brain_class() #doesnt work directly from pysurfer

    brain = Brain("fsaverage", **plot_kwargs)

    #mask locations based on percentile
    for hemi in ["lh", "rh"]:

        annot_file = subjects_dir + f'/fsaverage/label/{hemi}.{parc}.annot'
        labels, _, nib_names = nib.freesurfer.read_annot(annot_file)

        names_order_nib = np.array([str(name)[2:-1] for name in nib_names])

        if hemi == "lh":
            names_mne = names_order_mne[lh]
            cur_stc = stc_parc[lh]#, tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[lh]
        else:
            names_mne = names_order_mne[rh]
            cur_stc = stc_parc[rh]#, tmin:tmax].mean(axis=1)
            cur_mask = stc_mask[rh]

        # Create a dictionary to map strings to their indices in array1
        index_dict = {value: index for index, value in enumerate(names_mne)}

        # Find the indices of strings in array1 corresponding to array2
        right_order = [index_dict[value] for value in names_order_nib]

        cur_stc_ordered = cur_stc[right_order]
        cur_mask_ordered = cur_mask[right_order]
        
        cur_stc_ordered[cur_mask_ordered] = np.nan

        vtx_data = cur_stc_ordered[labels]
        vtx_data[labels == -1] = -1

        brain.add_data(vtx_data, hemi=hemi, fmin=stc_parc.min(), #fmid=1,
                       fmax=stc_parc.max(), 
                       colormap=cmap, #np.nanmax(stc_parc)
                       colorbar=False, alpha=.8)

    
    screenshot = brain.screenshot()
    #brain.close()
    
    return screenshot


#%%
eff_brain = plot_parc(stc_parc, stc_mask, 'magma')

#%%
df_knee_across = pd.read_csv(join(home_base, 'data/knee_across_brain.csv'))

knee_proba = np.concatenate([[df_knee_across['p(knee > fixed)'][0]], df_knee_across['p(knee > fixed)'].to_numpy()])

#%%
eff_brain = plot_parc(knee_proba, np.ones(np.shape(knee_proba)) == 0, 'RdBu_r')

#%%
eff_brain = plot_parc(knee_proba, knee_proba > 0.5, 'Reds')

#%%
eff_brain = plot_parc(knee_proba, knee_proba < 0.5, 'Reds')

#%%

fig, ax = plt.subplots()
ax.axis("off")

plt.imshow(eff_brain, cmap='RdBu_r')

cbaxes = inset_axes(plt.gca(), width="6%", height="36%", loc=7, borderpad=-2)
cbar = plt.colorbar(cax=cbaxes, ax=ax, orientation='vertical')
cbar.set_label('weights (scaled)')#, rotation=270)
plt.clim(0, 1)
#plt.tight_layout()
plt.show()
fig.tight_layout()

fig.savefi