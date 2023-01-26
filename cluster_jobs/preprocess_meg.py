#%%
from cluster_jobs.meta_job import Job
#%%
import mne
from os.path import join
import numpy as np
import joblib
import pandas as pd
import yasa
import eelbrain as eb


import sys
sys.path.append('/mnt/obob/staff/fschmidt/resting_tinnitus/')
from utils.cleaning_utils import run_potato, interpolate_blinks


import random
random.seed(42069) #make it reproducible - sort of

#%%

class PreprocessingJob(Job):

    job_data_folder = 'data_meg_05'

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


        #%% debug
        # subject_id = '19950901ptal'
        # l_pass = None
        # h_pass = 0.1
        # notch = False
        # eye_threshold = 0.5
        # heart_threshold = 0.5
        # powerline = 50 #in hz
        # duration=2
        # potato=False
        # freq_range = [0.1, 45]
        # meg_type='mag'
        # sss=True
        # pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True}


        data_frame_path = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/subject_lists/df_tinnitus.csv'

        df_all = pd.read_csv(data_frame_path).query('fs_1k == True')
        df_all.reset_index(inplace=True)
        df = df_all.query(f'subject_id == "{subject_id}"')
        cur_path = list(df['path'])[0]

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

        #% if time is below 5mins breaks function here -> this is because some people in salzburg recorded ~1min resting states
        if raw.times.max() / 60 < 4.9:
            raise ValueError(f'The total duration of the recording is below 5min. Recording duration is {raw.times.max() / 60} minutes')
        
        #% make sure that if channels are set as bio that they get added correctly
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
    
        #%
        # reject components by explained variance
        # find which ICs match the EOG pattern using correlation
        eog_indices, eog_scores = ica.find_bads_eog(raw_copy, measure='correlation', threshold=eye_threshold)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_copy, measure='correlation', threshold=heart_threshold)
    
        #% drop physiological components
        ica.apply(raw, exclude=ecg_indices + eog_indices)

        #%% run irasa on raw continuous data
        kwargs_welch_raw = {'average': 'mean',
                            'window': 'hann',
                            'noverlap': raw.info['sfreq']*duration/2}
        raw_mags = raw.copy().pick_types(meg=meg_type)

        raw_irasa = self._run_irasa(raw_mags, freq_range, duration, kwargs_welch_raw)

        #%% run on epochs
        #TODO: develop smart method to get vertical and horizontal eye movements       
        eog_events = np.concatenate([mne.preprocessing.find_eog_events(raw, ch_name=cur_eog) for cur_eog in ['EOG001', 'EOG002']])

        blink_epochs = mne.Epochs(raw, eog_events, picks='mag', event_id=998, tmin=-0.2, tmax=duration-0.2,
                                  baseline=(-0.2, -0.05), event_repeated='merge')

        epoch_data = blink_epochs.get_data(picks=meg_type)
        raw_new = mne.io.RawArray(np.hstack(epoch_data), raw_mags.info)


        kwargs_welch_epo = {'average': 'mean',
                            'window': 'hann',
                            'noverlap': 0}

        epoch_irasa = self._run_irasa(raw_new, freq_range, duration, kwargs_welch_epo)


        #%% eye event trf
        #TODO: Encoding model eye movement, eye events brain data
        raw_resample = raw.resample(sfreq=100)
        eog_event_list = [mne.preprocessing.find_eog_events(raw_resample, ch_name=cur_eog) for cur_eog in ['EOG001', 'EOG002']]
        eog_events = np.concatenate(eog_event_list)

        eog = raw_resample.get_data(picks='eog')
        mag4trfs = raw_resample.get_data(picks='mag')

        # get eog events
        eye_events = np.zeros(eog.shape[1]) 
        eye_events[eog_events[:,0] - raw_resample.first_samp] = 1

        # interpolate eog events
        my_eog_events = [cur_events[:,0] - raw_resample.first_samp for cur_events in eog_event_list]
        eog_clean = [interpolate_blinks(cur_eog, cur_eog_events, raw_resample.info['sfreq'], 250, 50) for cur_eog, cur_eog_events in zip(eog, my_eog_events)]
        
        # turn into ndvars
        #time axis
        tstep = 1. /raw_resample.info['sfreq']
        n_times = eog_clean[0].shape[0]
        time = eb.UTS(0, tstep, n_times)

        #eog data
        veog = eb.NDVar(eog_clean[0], (time, ), name='vEOG')
        heog = eb.NDVar(eog_clean[1], (time, ), name='hEOG')
        eye_eve = eb.NDVar(eye_events, (time, ), name='blinks/saccades')

        #meg data
        sensor = eb.load.fiff.sensor_dim(raw_resample, connectivity='grid')
        meg_trf = eb.NDVar(mag4trfs.T, (time, sensor), name='MEG', info={'unit': 'fT'})

        #% run boosting
        veog_trf = eb.boosting(meg_trf, veog, tstart=-.2, tstop=.4, basis=0.05)
        heog_trf = eb.boosting(meg_trf, heog, tstart=-.2, tstop=.4, basis=0.05)
        eye_eve_trf = eb.boosting(meg_trf, eye_eve, tstart=-.2, tstop=.4, basis=0.05)
        #% cmb boosting
        cmb_trf = eb.boosting(meg_trf, [veog, heog, eye_eve], tstart=-.2, tstop=.4, basis=0.05)
        
        #%%
        data = {'age': df['measurement_age'],
                'tinnitus': df['tinnitus'],
                'veog_trf': veog_trf,
                'heog_trf': heog_trf,
                'eye_eve_trf': eye_eve_trf,
                'cmb_trf': cmb_trf,
                'eog_scores': eog_scores,
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