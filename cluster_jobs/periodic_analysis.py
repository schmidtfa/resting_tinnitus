#%%
from cluster_jobs.meta_job import Job
#%%
from pathlib import Path
import joblib

from pyrasa.utils.peak_utils import get_band_info

import numpy as np
import pandas as pd

#%%

class peak_rasa(Job):

    job_data_folder = 'pyrasa_peak_params_duration_2'

    def run(self,
            subject_id,
            peak_threshold=1,
            min_peak_height=0.1,
            peak_width_limits=(0.5, 12),
            duration = 2
            ):

        #%%debug
        # subject_id = '19981024mrfr'#
        # peak_threshold=1
        # min_peak_height=0.1
        # peak_width_limits=(0.5, 12)

        # %%
        query_string = f'__duration_{duration}__hmax_2__source_surface__atlas_glasser.dat'
        INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'
        cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}' + query_string))[0]))

        #%% src based analysis

        peak_setup = {'peak_threshold': peak_threshold,
                        'min_peak_height': min_peak_height,
                        'peak_width_limits': peak_width_limits,
                    }
        
        def add_info(df, df_info):
            df['subject_id'] = df_info['subject_id'].iloc[0]
            df['tinnitus'] = df_info['tinnitus'].iloc[0]
            df['dB'] = df_info['dB'].iloc[0]
            df['age'] = df_info['measurement_age'].iloc[0]
            df['tinnitus_distress'] = df_info['tinnitus_distress'].iloc[0]

            return df

        labels = cur_data['src']['label_info']['names_order_mne']
        cur_data['src']['irasa_label_stc'].ch_names = labels
        peaks = add_info(cur_data['src']['irasa_label_stc'].get_peaks(**peak_setup), cur_data['subject_info'])

        
        ch_names, counts = np.unique(peaks['ch_name'], return_counts=True)

        chs_not_in_list = [label for label in labels if label not in ch_names]
        df_empty = pd.DataFrame({'ch_name': chs_not_in_list,
                                 'n_peaks': np.zeros(len(chs_not_in_list))})
        
        df_peaks = pd.concat([df_empty, pd.DataFrame({'ch_name': ch_names,
                                          'n_peaks': counts})])

        df_peaks = add_info(df_peaks, cur_data['subject_info']) 

        #%% 
    
        delta_ctx = add_info(get_band_info(peaks, freq_range=(0, 3), 
                                        ch_names=labels), cur_data['subject_info']) 
        theta_ctx = add_info(get_band_info(peaks, freq_range=(3, 7), 
                                        ch_names=labels), cur_data['subject_info']) 
        alpha_ctx = add_info(get_band_info(peaks, freq_range=(7, 14), 
                                        ch_names=labels), cur_data['subject_info']) 
        beta_ctx = add_info(get_band_info(peaks, freq_range=(15, 30), 
                                        ch_names=labels), cur_data['subject_info']) 
        gamma_ctx = add_info(get_band_info(peaks, freq_range=(30, 45), 
                                        ch_names=labels), cur_data['subject_info']) 
        
        data = {'peaks': peaks,
                'bands': {'delta': delta_ctx,
                                'theta': theta_ctx,
                                'alpha': alpha_ctx,
                                'beta': beta_ctx,
                                'gamma': gamma_ctx,
                                },
                'df_peaks': df_peaks}    

        #%%
        #TODO: ADD ECG and EOG analysis

        joblib.dump(data, self.full_output_path)