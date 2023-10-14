#%%
import mne
from os.path import join
import pickle
import pandas as pd


subs2check = ['19640925tewm', '19571220rsdo']

#%%
home_base = '/Users/b1059770/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes.noindex/bomber/resting_tinnitus'

mri_path = join(home_base, 'data/freesurfer')
fs_path = join(mri_path, 'fsaverage')
trans = 'fsaverage' 


plot_kwargs = dict(subject=trans, subjects_dir=mri_path,
                   surfaces="head-dense", dig=True, eeg=[],
                   meg='sensors', show_axes=True,
                   coord_frame='meg')
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                   focalpoint=(0., 0., 0.))

hmfolder = join(home_base, 'data/headmodels/')

df_all = pd.read_csv(join(home_base, 'data/tinnitus_match.csv'))
subject_ids = list(df_all['subject_id'].unique())


subID = subs2check[1]

coreg  = mne.read_trans(hmfolder + subID + '/' + subID + '-trans.fif')

infofile = open(hmfolder + subID + '/info.pickle', "rb")
info = pickle.load(infofile)


#%%
fig = mne.viz.plot_alignment(info, trans=coreg, **plot_kwargs) #CHECK
fig.plotter.add_text('Subject ID: ' + subID, color='black')
mne.viz.set_3d_view(fig, **view_kwargs)
# %%
mne.io.read_info(hmfolder + subID + '/' + subID + '-trans.fif')

