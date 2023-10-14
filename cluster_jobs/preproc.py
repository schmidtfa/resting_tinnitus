#%%
from cluster_jobs.meta_job import Job
#%%
import joblib
import pandas as pd
import scipy.signal as dsp
from neurodsp.spectral import compute_spectrum_welch

import sys
sys.path.append('/mnt/obob/staff/fschmidt/resting_tinnitus/utils')
from preproc_utils import preproc_data
from src_utils import raw2source
import mne
import numpy as np
from os.path import join



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
            sgramm=False,
            downsample_f = 1000, #make sure that the 10 or 5k data is also at 1k
            ica_threshold = 0.5,
            duration=4):


        #%% debug
        # subject_id = '19891222gbhl'
        # l_pass = None
        # h_pass = 1
        # notch = False
        # do_ica = False
        # ica_threshold = 0.5
        # max_filt=False
        # downsample_f = None
        # duration=4
        # sgramm=True,

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
        
        #% crop data (if it exceeds 5min -> typical this is not the case only for some p10 subjects we did 15min recordings)
        if raw.times.max() / 60 > 5:
                raw.crop(0, 5*60)

        eog = raw.get_data(picks='eog')
        ecg = raw.get_data(picks='ecg')

        fs = raw.info['sfreq']
        welch_settings = {'fs': fs, 
                        'avg_type':'median',
                        'window': 'hann',
                        'nperseg': fs*duration,
                        'noverlap': fs*duration / 2, #50% overlap as default
                        'outlier_percent': 5}

        #%% move data to source
        #adjust preprocessing settings for empty room
        preproc_settings['coord_frame'] = 'meg'
        preproc_settings['do_ica'] = False
        subjects_dir = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/freesurfer'

        stc = raw2source(raw, subject_id, subjects_dir, preproc_settings)

        #% get tc from parcellation and return
        fs_path = join(subjects_dir, 'fsaverage')
        src_file = join(fs_path, 'bem', 'fsaverage-ico-4-src.fif')
        src = mne.read_source_spaces(src_file)
        labels_mne = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', subjects_dir=subjects_dir)

        names_order_mne = np.array([label.name[:-3] for label in labels_mne])

        rh = [True if label.hemi == 'rh' else False for label in labels_mne]
        lh = [True if label.hemi == 'lh' else False for label in labels_mne]

        label_info = {'lh': lh,
                      'rh': rh,
                      'parc': 'HCPMMP1',
                      'names_order_mne': names_order_mne}
        
        if sgramm:
                sgramm_settings = {'fs':fs,
                                   'window': 'hann',
                                   'nperseg': int(fs*duration), 
                                   'noverlap': int(fs*duration/2)}
                freq, _, eog_psd = dsp.spectrogram(eog, **sgramm_settings)
                _, _, ecg_psd = dsp.spectrogram(ecg, **sgramm_settings)

                #slight deviation from below. I need to extract the label_tc before doing fft.
                #Otherwise i would need to hack mne more than i want to
                label_tc = mne.extract_label_time_course(stc, labels_mne, src, mode='mean_flip') #TODO: Maybe try PCA

                data_dict = {'label_tc': label_tc,
                             'label_info': label_info}
                
                _, _, data_dict['label_tc'] = dsp.spectrogram(label_tc, **sgramm_settings)

        else:
                freq, eog_psd = compute_spectrum_welch(eog, **welch_settings)
                _, ecg_psd = compute_spectrum_welch(ecg, **welch_settings)
                _, stc.data = compute_spectrum_welch(stc.data, **welch_settings) 

                label_tc = mne.extract_label_time_course(stc, labels_mne, src, mode='mean')

                data_dict = {'label_tc': label_tc,
                             'label_info': label_info}


        #%%
        data = {'subject_info': df,
                'eog': eog_psd,
                'ecg': ecg_psd,
                'src': data_dict,
                'freq': freq,
                'subject_id': subject_id,
                }

        joblib.dump(data, self.full_output_path)