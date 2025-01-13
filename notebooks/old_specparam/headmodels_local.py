#%%
import numpy as np
import mne
from mne.coreg import Coregistration
from os.path import join
import pandas as pd
import pickle

bad_subjects = ['20000216rpgu', #10khz
                '19940909gbkr', 
                '19620826wlti',
                '19480615kthn', 
                '19650312eips', #head is tilted
                '19620430grgc',
                '19991224hika',
                '19641104usbl']

local = True

subject_id = bad_subjects[6]
maxf =False

if local:
    home_base = '/Users/b1059770/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes.noindex/bomber/resting_tinnitus'
else:
    home_base = '/mnt/obob/staff/fschmidt/resting_tinnitus/'

#%%
mri_path = join(home_base, 'data/freesurfer')
fs_path = join(mri_path, 'fsaverage')
trans = 'fsaverage' 

plot_kwargs = dict(subject=trans, subjects_dir=mri_path,
                   surfaces="head-dense", dig=True, eeg=[],
                   meg='sensors', show_axes=True,
                   coord_frame='meg')
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                   focalpoint=(0., 0., 0.))

df = pd.read_csv(join(home_base, 'data/tinnitus_match.csv')).query(f'subject_id == "{subject_id}"')

path_sinuhe = '/Users/b1059770/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes.noindex/obob'

#%%
raw = mne.io.read_raw_fif(join(path_sinuhe, df['path'].to_list()[0][1:]), verbose=False)


if maxf:
    max_settings_path = join(home_base, 'utils')
    #cal & cross talk files specific to system
    calibration_file = join(max_settings_path, 'sss_cal.dat')
    cross_talk_file = join(max_settings_path, 'ct_sparse.fif')
            
    #find bad channels first
    noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw,
                                                                        calibration=calibration_file,
                                                                        cross_talk=cross_talk_file)
    #Load data
    raw.load_data()
    raw.info['bads'] = noisy_chs + flat_chs

    raw = mne.preprocessing.maxwell_filter(raw,
                                            calibration=calibration_file,
                                            cross_talk=cross_talk_file,
                                            destination=None,  # noqa
                                            st_fixed=False
                                            )


info = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=True))

#%% do the coregistration
coreg = Coregistration(info, trans, mri_path)
coreg.fit_fiducials(verbose=True)


#%%
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs) 
fig.plotter.add_text('Subject ID: ' + subject_id, color='black')
mne.viz.set_3d_view(fig, **view_kwargs)

#%%
coreg.fit_icp(n_iterations=6, nasion_weight=2, verbose=True)
coreg.omit_head_shape_points(distance=15 / 1000)

#%%
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs) 
fig.plotter.add_text('Subject ID: ' + subject_id, color='black')
mne.viz.set_3d_view(fig, **view_kwargs)

#%%
coreg.fit_icp(n_iterations=20, nasion_weight=10, verbose=True)

dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
print(f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "f"/ {np.min(dists):.2f} mm / "
    f"{np.max(dists):.2f} mm")

#%%
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs) 
fig.plotter.add_text('Subject ID: ' + subject_id, color='black')
mne.viz.set_3d_view(fig, **view_kwargs)


#%% save

file = open(join(home_base, "data/headmodels/") + subject_id + "/info.pickle", 'wb')
pickle.dump(info, file)
file.close()    

outpath = join(home_base, f'data/headmodels/{subject_id}/{subject_id}-trans.fif')

mne.write_trans(outpath, coreg.trans, overwrite=True)
print('Coregistration done!')
# %%
