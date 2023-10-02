import pyriemann
from scipy.signal import savgol_filter
import numpy as np

def run_potato(epochs, potato_threshold=2):

    '''
    This function takes an mne.epochs object and identifies outlying ("bad") segments using some riemann magic 
    '''

    # estimate the covariance matrices from our epochs
    covs = pyriemann.estimation.Covariances(estimator="lwf")
    cov_mats = covs.fit_transform(epochs.get_data())
        
    # Fit the Potato
    potato = pyriemann.clustering.Potato(threshold=potato_threshold)
    potato.fit(cov_mats)

    # Get the clean epoch indices and select only these for further processing
    clean_idx = potato.predict(cov_mats).astype(bool)
    return epochs[clean_idx]  


        

def interpolate_blinks(data, eog_events, fs, blink_intervall, intervall_pre):

    blink_intervall = int(fs / 1000 * blink_intervall) #get ms in samples
    intervall_pre = int(fs / 1000 * intervall_pre)

    for event in eog_events:

        cur_onset = event - (blink_intervall + intervall_pre)
        cur_offset = event + (blink_intervall + intervall_pre)

        if cur_onset > 0 and cur_offset < len(data):

            start_blink_val, stop_blink_val = data[cur_onset], data[cur_offset]
            data[cur_onset:cur_offset] = np.linspace(start_blink_val, stop_blink_val, num = cur_offset - cur_onset)
            data[cur_onset-intervall_pre:cur_offset+intervall_pre] = savgol_filter(data[cur_onset-intervall_pre:cur_offset+intervall_pre], 11, 2)

        else:
            print('Not enough data for interpolation')


    return data