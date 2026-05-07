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

    job_data_folder = 'pyrasa_peak_params_duration_6_final_default'

    def run(self,
            subject_id,
            # peak_threshold=1,
            # min_peak_height=0.1, #set to 0.1 in actual analysis
            # peak_width_limits=(1, 6),
            # min_peak_distance_hz=2,
            #smoothing_window=1,
            peak_threshold=2., 
            #peak_width_limits=[0.5, 8],
            smoothing_window=2, #keep as is
            # peak_threshold=2.,
            # min_peak_height=0.01,
            peak_width_limits=[1, 6], #keep as is
            duration = 6
            ):

        #%%debug
        # subject_id = '19600107brgt'#
        # peak_threshold=1
        # min_peak_height=0.1
        # peak_width_limits=(0.5, 12)
        # duration=2

        # %%
        query_string = f'__duration_{duration}__hmax_2__source_surface__atlas_glasser.dat'
        INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'
        cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}' + query_string))[0]))

        #%% src based analysis
                                        

        peak_setup = {'peak_threshold': peak_threshold,
                      'smoothing_window':smoothing_window,
                      'peak_width_limits': peak_width_limits,
        #               'min_peak_height':min_peak_height,
        # #                 'min_peak_distance_hz':min_peak_distance_hz,
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
        def get_band_info(df_peaks: pd.DataFrame, freq_range: tuple[int, int], ch_names: list) -> pd.DataFrame:
            """
            Extract peak information within a specified frequency range from a DataFrame of peak parameters.

            This function filters peaks found in the periodic spectrum or spectrogram to those within
            a specified frequency range.It ensures that every channel is represented in the output,
            filling in missing channels (i.e., channels without detected peaks in the specified range) with NaN values.

            Parameters
            ----------
            df_peaks : pd.DataFrame
                DataFrame containing peak parameters obtained from the `get_peak_params` function.
                The DataFrame should include columns for 'ch_name' (channel name), 'cf' (center frequency),
                'bw' (bandwidth), and 'pw' (peak height).
            freq_range : tuple of (int, int)
                Tuple specifying the lower and upper frequency bounds (in Hz) to filter peaks by. Only peaks
                with center frequencies (cf) within this range will be included in the output.
            ch_names : list
                List of channel names used in the computation of the periodic spectrum. This list ensures that
                every channel is accounted for in the output, even if no peaks were found in the specified range
                for certain channels.

            Returns
            -------
            pd.DataFrame
                DataFrame containing the peak parameters ('cf', 'bw', 'pw') for each channel within the specified
                frequency range. Channels without detected peaks in this range will have NaN values for these parameters.
                The DataFrame includes:
                - 'ch_name': Channel name
                - 'cf': Center frequency of the peak within the specified range
                - 'bw': Bandwidth of the peak within the specified range
                - 'pw': Peak height (power) within the specified range

            Notes
            -----
            This function is useful for isolating and analyzing peaks that occur within specific canonical frequency bands
            (e.g., alpha, beta, gamma) across multiple channels in a periodic spectrum. The inclusion of NaN
            entries for channels without detected peaks ensures that the output DataFrame is complete and aligned
            with the original channel list.

            """

            df_range = df_peaks.query(f'cf >= {freq_range[0]}').query(f'cf <= {freq_range[1]}') #make it inclusive

            # we dont always get a peak in a queried range lets give those channels a nan
            missing_channels = list(set(ch_names).difference(df_range['ch_name'].unique()))
            missing_df = pd.DataFrame(np.nan, index=np.arange(len(missing_channels)), columns=['ch_name', 'cf', 'bw', 'pw'])
            missing_df['ch_name'] = missing_channels

            df_band_peaks = pd.concat([df_range, missing_df]).reset_index().drop(columns='index')

            return df_band_peaks


    
        delta_ctx = add_info(get_band_info(peaks, freq_range=(1, 3), 
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

        joblib.dump(data, self.full_output_path)