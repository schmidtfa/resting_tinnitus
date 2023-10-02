#%%
import numpy as np
import mne
from mne.coreg import Coregistration
from os.path import join
import pandas as pd
from cluster_jobs.meta_job import Job
import pickle

#%%
class HeadModelJob(Job):
    
    job_data_folder = 'headmodels'
    data_file_suffix = '-trans.fif'
    
    def run(self, 
            subject_id):
      
        mri_path = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/freesurfer'
        fs_path = join(mri_path, 'fsaverage')
        trans = 'fsaverage' 
        
        df = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv').query(f'subject_id == "{subject_id}"')

        raw = mne.io.read_raw_fif(df['path'].to_list()[0], verbose=False)

        max_settings_path = '/mnt/obob/staff/fschmidt/resting_tinnitus/utils'
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
        coreg.fit_icp(n_iterations=6, nasion_weight=2, verbose=True)
        coreg.omit_head_shape_points(distance=15 / 1000)
        coreg.fit_icp(n_iterations=20, nasion_weight=10, verbose=True)

        dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
        print(f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "f"/ {np.min(dists):.2f} mm / "
            f"{np.max(dists):.2f} mm")

        file = open("/mnt/obob/staff/fschmidt/resting_tinnitus/data/headmodels/" + subject_id + "/info.pickle", 'wb')
        pickle.dump(info, file)
        file.close()    
        
        mne.write_trans(self.full_output_path, coreg.trans, overwrite=True)
        print('Coregistration done!')


# %% UNCOMMENT FOR TESTING
if __name__ == '__main__':

    subject_id = '19670901igsr'

    job = HeadModelJob(subject_id=subject_id)
    job.run_private()

# %%