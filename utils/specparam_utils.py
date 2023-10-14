import pandas as pd
import numpy as np
from fooof.analysis import get_band_peak_fg
from fooof.utils.params import compute_knee_frequency



def get_band_peak_df(fg, bands, ch_names):

    """Extract peaks in preset frequency bands and return a pandas dataframe"""

    df_list = []
    for ix, ch in enumerate(ch_names):
        #TODO: this is super inefficient -> improve
        cur_peak_df = pd.DataFrame({cur_band : get_band_peak_fg(fg, bands[cur_band])[ix] for cur_band in bands.labels})
        cur_peak_df['peak_params'] = ['cf', 'pw', 'bw']
        cur_peak_df['ch_name'] = ch
        cur_peak_df['n_peaks'] = fg.n_peaks_[ix]
        cur_peak_df['r_squared'] = fg.get_params('r_squared')[ix]
        cur_peak_df['error'] = fg.get_params('error')[ix]
        df_list.append(cur_peak_df)
    
    df_peaks = pd.concat(df_list)
    return df_peaks



def get_aperiodic(fg, ch_names, key, aperiodic_mode):

    """Extract aperiodic activity and return a pandas dataframe"""

    if aperiodic_mode == 'knee':
        df_ap = pd.DataFrame(fg.get_params('aperiodic_params'), columns=['offset', 'knee', 'exponent'])
        df_ap['knee_freq'] = compute_knee_frequency(df_ap['knee'], df_ap['exponent'])

    else:

        df_ap = pd.DataFrame(fg.get_params('aperiodic_params'), columns=['offset', 'exponent'])
        df_ap['knee'] = np.nan
        df_ap['knee_freq'] = np.nan

    
    df_ap['r_squared'] = fg.get_params('r_squared')
    df_ap['error'] = fg.get_params('error')
    df_ap['n_peaks'] = fg.n_peaks_

    if key == 'ecg':
        df_ap['ch_name'] = 'ECG'
        df_ap = df_ap.head(1) #take only the "first" ecg channel
    else:
        df_ap['ch_name'] = ch_names
    return df_ap





def knee_or_fixed(cur_df):

    def _calculate_bic(n, error, num_params):

        """Calculates BIC using the model error from fooof"""

        bic = n * np.log(error) + num_params * np.log(n)
        return bic

    freqs = np.arange(0.5, 100, 0.25).shape[0] #0.5 to 100hz in steps of .25
    cur_df['knee_modeled'] = (cur_df['aperiodic_mode'] == 'knee').astype(int)
    cur_df['num_params'] = cur_df['knee_modeled'] + cur_df['n_peaks']
    cur_df['bic_ap_mode'] = _calculate_bic(freqs, cur_df['error'], cur_df['num_params'])

    df_new = []

    for subject in cur_df['subject_id'].unique():

        cur_ap = cur_df.query(f'subject_id == "{subject}"')

        k = cur_ap.query('aperiodic_mode == "knee"')
        f = cur_ap.query('aperiodic_mode == "fixed"')

        #Note: this works because channels are sorted the same (but its a bit dangerous)
        #check for nans as they may result in dropped channels
        #print(subject)
        knee_chs = list(f['ch_name'][f['bic_ap_mode'] > k['bic_ap_mode']]) #knee better (smaller bic wins)
        fixed_chs = list(f['ch_name'][f['bic_ap_mode'] < k['bic_ap_mode']])

        df_new.append(pd.concat([k.query('ch_name == @knee_chs'), f.query('ch_name == @fixed_chs')]))

    return  pd.concat(df_new)
