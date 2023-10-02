import pandas as pd
import numpy as np



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
        knee_chs = list(f['ch_name'][f['bic_ap_mode'] > k['bic_ap_mode']]) #knee better (smaller bic wins)
        fixed_chs = list(f['ch_name'][f['bic_ap_mode'] < k['bic_ap_mode']])

        df_new.append(pd.concat([k.query('ch_name == @knee_chs'), f.query('ch_name == @fixed_chs')]))

    return  pd.concat(df_new)
