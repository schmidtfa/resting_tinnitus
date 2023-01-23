#%%
from cluster_jobs.meta_job import Job
import mne
from os.path import join
import numpy as np
import joblib
import pandas as pd
import yasa
from fooof.utils import interpolate_spectrum


import sys
sys.path.append('/mnt/obob/staff/fschmidt/resting_tinnitus/')
from utils.cleaning_utils import run_potato
from utils.psd_utils import compute_spectra_ndsp, compute_spectra_mne, interpolate_line_freq
from utils.fooof_utils import fooof2aperiodics


import random
random.seed(42069) #make it reproducible - sort of

#%%

class PreprocessingJob(Job):

    job_data_folder = 'data_meg'

    def run(self,
            subject_id,
            l_pass = None,
            h_pass = 0.1,
            notch = False,
            eye_threshold = 0.5,
            heart_threshold = 0.5,
            powerline = 50, #in hz
            duration=2,
            potato=False,
            freq_range = [0.1, 45],
            meg_type='mag',
            sss=True,
            pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True}):


        data_frame_path = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/subject_lists/df_tinnitus.csv'

        df_all = pd.read_csv(data_frame_path).query('fs_1k == True')
        df_all.reset_index(inplace=True)
        df = df_all.query(f'subject_id == "{subject_id}"')
        cur_path = list(df['path'])[0]

        #%%

        raw = mne.io.read_raw_fif(cur_path)

        max_settings_path = '/mnt/obob/staff/fschmidt/meeg_preprocessing/meg/maxfilter_settings/'
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

        if sss:
            raw = mne.preprocessing.maxwell_filter(raw,
						   calibration=calibration_file,
                                                   cross_talk=cross_talk_file,
                                                   destination=(0, 0, 0.04),  # noqa
                                                   st_fixed=False
                                                   )
        else:
            raw.interpolate_bads()

        #%% if time is below 5mins breaks function here -> this is because some people in salzburg recorded ~1min resting states
        if raw.times.max() / 60 < 4.9:
            raise ValueError(f'The total duration of the recording is below 5min. Recording duration is {raw.times.max() / 60} minutes')
        
        #%% make sure that if channels are set as bio that they get added correctly
        if 'BIO003' in raw.ch_names:
            raw.set_channel_types({'BIO001': 'eog',
                                   'BIO002': 'eog',
                                   'BIO003': 'ecg',})

            mne.rename_channels(raw.info, {'BIO001': 'EOG001',
                                           'BIO002': 'EOG002',
                                           'BIO003': 'ECG003',})

        raw.pick_types(**pick_dict)
        #Apply filters
        raw.filter(l_freq=h_pass, h_freq=l_pass)

        if notch:
            nyquist = raw.info['sfreq'] / 2
            print(f'Running notch filter using {powerline} Hz steps. Nyquist is {nyquist}')
            raw.notch_filter(np.arange(powerline, nyquist, powerline), filter_length='auto', phase='zero')
            
        #Do the ica
        print('Running ICA. Data is copied and the copy is high-pass filtered at 1Hz')
        raw_copy = raw.copy().filter(l_freq=1, h_freq=None)
    
        ica = mne.preprocessing.ICA(n_components=50, #selecting 50 components here -> fieldtrip standard in our lab
                                    max_iter='auto')
        ica.fit(raw_copy)
        ica.exclude = []
    
        #%%
        # reject components by explained variance
        # find which ICs match the EOG pattern using correlation
        eog_indices, eog_scores = ica.find_bads_eog(raw_copy, measure='correlation', threshold=eye_threshold)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_copy, measure='correlation', threshold=heart_threshold)
    
        #%% select everything but heart stuff
        ica.apply(raw, exclude=ecg_indices + eog_indices)

        #%% on raw
        kwargs_welch_raw = {'average': 'mean',
                            'window': 'hann',
                            'noverlap': raw.info['sfreq']*duration/2}
        raw_mags = raw.copy().pick_types(meg=meg_type)


        raw_irasa = self._run_irasa(raw_mags, freq_range, duration, kwargs_welch_raw)

        #%% on epochs
        #TODO: develop smart method to get vertical and horizontal eye movements       
        eog_events = np.concatenate([mne.preprocessing.find_eog_events(raw, ch_name=cur_eog) for cur_eog in ['EOG001', 'EOG002']])

        blink_epochs = mne.Epochs(raw_mags, eog_events, event_id=998, tmin=-0.2, tmax=duration-0.1, baseline=(-0.2, -0.05))

        #%%
        epoch_data = blink_epochs.get_data()

        #%%
        if meg_type == 'mag':
            raw_new = mne.io.RawArray(epoch_data.reshape(102, -1), raw_mags.info)

        elif meg_type == 'grad':
            raw_new = mne.io.RawArray(epoch_data.reshape(204, -1), raw_mags.info)

        kwargs_welch_epo = {'average': 'mean',
                            'window': 'hann',
                            'noverlap': 0}

        epoch_irasa = self._run_irasa(raw_new, freq_range, duration, kwargs_welch_epo)
        
        #%%
        data = {'age': df['measurement_age'],
                'tinnitus': df['tinnitus'],
                'raw_irasa': raw_irasa,
                'epoch_irasa': epoch_irasa,
                'blink_epochs': blink_epochs,
                'subject_id': subject_id,
                }

        joblib.dump(data, self.full_output_path)


    def _run_irasa(self, cur_data, freq_range, duration, kwargs_welch):

        freqs, psd_aperiodic, psd_osc, fit_params = yasa.irasa(cur_data, band=freq_range, win_sec=duration, kwargs_welch=kwargs_welch) #50% overlap

        irasa_data = {
            'aperiodic': psd_aperiodic,
            'periodic':psd_osc,
            'freqs': freqs,
            'fit_params': fit_params
        }
        return irasa_data