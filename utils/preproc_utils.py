import mne
import numpy as np
from datetime import datetime
from os import listdir
from os.path import join
from rm_train import rm_train_ica


def preproc_data(cur_data, 
                 max_filt=True,
                 notch=False,
                 coord_frame='head',
                 l_pass=None,
                 h_pass=None,  
                 do_ica=False,
                 ica_threshold=0.5,
                 downsample_f=None):
    
    """
    Minimal preprocessing function. Only contains some maxfiltering.
    I might add an automatic ICA and autoreject, riemannian potato option
    """

    if isinstance(cur_data, str):
        cur_data = mne.io.read_raw_fif(cur_data, preload=True, verbose=False, on_split_missing='warn')

    if max_filt:
        print('Running maxfilter')
        calibration_file = '/mnt/obob/staff/fschmidt/resting_tinnitus/utils/sss_cal.dat'
        cross_talk_file = '/mnt/obob/staff/fschmidt/resting_tinnitus/utils/ct_sparse.fif'

        # find bad channels first
        noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(cur_data,
                                                                          coord_frame=coord_frame,
                                                                          calibration=calibration_file,
                                                                          cross_talk=cross_talk_file  # noqa
                                                                          )
        cur_data.info['bads'] = noisy_chs + flat_chs

        cur_data = mne.preprocessing.maxwell_filter(cur_data,
                                                    calibration=calibration_file,
                                                    cross_talk=cross_talk_file,
                                                    coord_frame=coord_frame,
                                                    destination=None,  #NOTE: set to (0, 0, 0.04) if you want to compare sensor data
                                                    st_fixed=False)

        #% make sure that if channels are set as bio that they get added correctly
    if 'BIO003' in cur_data.ch_names:
        cur_data.set_channel_types({'BIO001': 'eog',
                                    'BIO002': 'eog',
                                    'BIO003': 'ecg',})

        mne.rename_channels(cur_data.info, {'BIO001': 'EOG001',
                                            'BIO002': 'EOG002',
                                            'BIO003': 'ECG003',})

    if np.logical_or(l_pass != None, h_pass != None):
        cur_data.filter(l_freq=h_pass, h_freq=l_pass)

    if notch:
        cur_data.notch_filter(np.arange(50, 351, 50), filter_length='auto', phase='zero')


    if do_ica:#Do the ica
        print('Running ICA. Data is copied and the copy is high-pass filtered at 1Hz')
        cur_data_copy = cur_data.copy().filter(l_freq=1, h_freq=None)

        ica = mne.preprocessing.ICA(n_components=50, #selecting 50 components here -> fieldtrip standard in our lab
                                    max_iter='auto')
        ica.fit(cur_data_copy)
        ica.exclude = []

        #%
        # reject components by explained variance
        # find which ICs match the EOG pattern using correlation
        eog_indices, _ = ica.find_bads_eog(cur_data_copy, measure='correlation', threshold=ica_threshold)
        ecg_indices, _ = ica.find_bads_ecg(cur_data_copy, measure='correlation', threshold=ica_threshold)

        #remove train
        r_train = rm_train_ica(cur_data_copy, ica)
        train_indices = list(np.arange(len(r_train))[r_train > ica_threshold])

        #% drop physiological components
        ica.apply(cur_data, exclude=ecg_indices + eog_indices + train_indices)

    if downsample_f != None:
        cur_data.resample(downsample_f, npad="auto")

    return cur_data