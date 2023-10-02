#%%
from cluster_jobs.meta_job import Job
#%%
import joblib
import pandas as pd
import scipy.signal as dsp
import sys
sys.path.append('/mnt/obob/staff/fschmidt/resting_tinnitus/utils')
from preproc_utils import preproc_data
from src_utils import raw2source


import random
random.seed(42069) #make it reproducible - sort of

#%%

class PreprocessingJob(Job):

    job_data_folder = 'data_meg'

    def run(self,
            subject_id,
            max_filt=True,
            l_pass = None,
            h_pass = 0.1,
            notch = False,
            do_ica = True,
            downsample_f = 1000, #make sure that the 10 or 5k data is also at 1k
            ica_threshold = 0.5,
            duration=4):


        #%% debug
        # subject_id = '19891222gbhl'
        # l_pass = None
        # h_pass = 1
        # notch = False
        # do_ica = True
        # ica_threshold = 0.5
        # max_filt=False
        # downsample_f = None
        # duration=4

        df = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv').query(f'subject_id == "{subject_id}"')
        cur_path = df['path'].to_list()[0]

        preproc_settings = {'max_filt': max_filt, 
                            'notch': notch,
                            'coord_frame':'head',
                            'l_pass': l_pass, 
                            'h_pass': h_pass, 
                            'do_ica': do_ica, 
                            'ica_threshold': ica_threshold,
                            'downsample_f': downsample_f}


        raw = preproc_data(cur_path, **preproc_settings)
        eog = raw.get_data(picks='eog')
        ecg = raw.get_data(picks='ecg')

        #%% move data to source
        #adjust preprocessing settings for empty room
        preproc_settings['coord_frame'] = 'meg'
        preproc_settings['do_ica'] = False
        subjects_dir = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/freesurfer'

        data_dict = raw2source(raw, subject_id, subjects_dir, preproc_settings, parc='HCPMMP1')

        #%% compute spectra on raw continuous data
        fs = raw.info['sfreq']
        welch_settings = {'fs': fs,
                          'window': 'hann',
                          'nperseg': fs*duration,
                          'noverlap': fs*duration / 2, #50% overlap as default
                          'detrend': 'constant',
                          'average': 'median'} #dont use mean if no bad epochs are rejected

        #sampling rate is similar and settings are the same so we jsut need the freqs once
        freq, eog_psd = dsp.welch(eog, **welch_settings)
        _, ecg_psd = dsp.welch(ecg, **welch_settings)
        _, data_dict['label_tc'] = dsp.welch(data_dict['label_tc'], **welch_settings) #overwrite to save some memory
        
        #%%
        data = {'subject_info': df,
                'eog': eog_psd,
                'ecg': ecg_psd,
                'src': data_dict,
                'freq': freq,
                'subject_id': subject_id,
                }

        joblib.dump(data, self.full_output_path)