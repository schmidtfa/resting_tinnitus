# %% fooof over time
# could do it with fooof_3d, 
# but i want to use subject and channel specific settings for knee or no knee fits

from cluster_jobs.meta_job import Job
#%%
from pathlib import Path
import joblib
from fooof import FOOOFGroup, Bands, fit_fooof_3d
from fooof.analysis import get_band_peak_fg
from fooof.utils.params import compute_knee_frequency
from fooof.utils import interpolate_spectrum

import numpy as np
import pandas as pd

import random
random.seed(42069) #make it reproducible - sort of

#%%

class SpecparamTime(Job):

    job_data_folder = 'specparam_temporal'

    def run(self,
            subject_id,
            aperiodic_mode,
            freq_range=(1, 98),
            max_n_peaks=6,
            do_bic=False,
            n_avgs = 5,
            peak_threshold=2,
            ):

        #%%debug
        # subject_id = '19480905mtbu'
        # aperiodic_mode = 'knee'
        # freq_range=(1, 98)
        # max_n_peaks=1
        # do_bic=False
        # n_avgs = 5

        INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
        cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__sgramm_True.dat'))[0])

        #define some bands
        bands = Bands({ 'delta' : [1, 3],
                        'theta' : [3, 8], 
                        'alpha' : [8, 14],
                        'beta' : [14, 30],
                        'gamma' : [30, 45],
                        'line_noise': [49, 51]})

        #%%
        periodic_list, aperiodic_list = [],[]

        for key in ['ecg', 'eog', 'src']:

            if key == 'eog':
                ch_names = ['EOGV', 'EOGH']
                data2fooof = cur_data[key]
            elif key == 'ecg':
                ch_names = ['ECG', 'ECG2']
                #NOTE: little trick to keep using fooofgroup -> only one channel is kept in later parts of the analysis
                data2fooof = np.vstack([cur_data[key], cur_data[key]]) 
            elif key == 'src':
                ch_names = cur_data['src']['label_info']['names_order_mne']
                data2fooof = cur_data['src']['label_tc']
        
            #% Do some subaveraging
            #data_shape = data2fooof.shape
            #residual = data_shape[2] - data_shape[2] % n_avgs
            #data2fooof = data2fooof[:,:,:residual].reshape(data_shape[0], data_shape[1], int(residual / n_avgs), n_avgs).mean(axis=-1)
            #%% better use a moving average over time
            def moving_average(x, w):
                return np.convolve(x, np.ones(w), 'valid') / w
            
            data2fooof = np.array([[moving_average(cur_ch[freq,:], w=n_avgs) for freq in range(cur_ch.shape[0])] for cur_ch in data2fooof])


            #%% do some spectral parametrization
            fg = FOOOFGroup(peak_width_limits=(0.5,4), #more doesnt really make sense to me
                            max_n_peaks=max_n_peaks, #this uses my fooof version with bic -> i.e. best model combination between parsimony and complexity is picked
                            aperiodic_mode=aperiodic_mode,
                            do_bic=do_bic,
                            peak_threshold=peak_threshold,
                            verbose=False) #otherwise log-files > 40gb
            
            fgs = fit_fooof_3d(fg, cur_data['freq'], np.swapaxes(data2fooof, 1, 2), freq_range=freq_range)

            #%
            for fg, ch_name in zip(fgs, ch_names):

                #% get band specific information
                df_periodic = self._get_band_peak_df(fg, bands, np.arange(data2fooof.shape[2]))
                #df_periodic = _get_band_peak_df(fg, bands, np.arange(data2fooof.shape[2]))
                df_periodic['ch_name'] = ch_name
                df_periodic['aperiodic_mode'] = aperiodic_mode

                #% get aperiodic information
                df_aperiodic = self._get_aperiodic(fg, np.arange(data2fooof.shape[2]), aperiodic_mode)
                #df_aperiodic = _get_aperiodic(fg, np.arange(data2fooof.shape[2]), aperiodic_mode)
                df_aperiodic['ch_name'] = ch_name
                df_aperiodic['aperiodic_mode'] = aperiodic_mode

                periodic_list.append(df_periodic)
                aperiodic_list.append(df_aperiodic)

        #%% combine all info in one df
        df_cmb_p = pd.concat(periodic_list).query('ch_name != "ECG2"')
        df_cmb_a = pd.concat(aperiodic_list).query('ch_name != "ECG2"')

        #add demographics
        demo_list = ['subject_id', 'tinnitus', 'measurement_age', 'dB','tinnitus_distress']
        sub_info = cur_data['subject_info'][demo_list].drop_duplicates('subject_id')

        data = {}
        for key, df in zip(['periodic', 'aperiodic'], [df_cmb_p, df_cmb_a]):

            df['subject_id'] = cur_data['subject_id']
            

            data.update({key: df.merge(sub_info, on='subject_id')})

        #save
        data['freq'] = cur_data['freq']
        data['label_info'] = cur_data['src']['label_info']

        joblib.dump(data, self.full_output_path)


    #%% utility functions
    def _get_band_peak_df(self,
                          fg, bands, times):

        """Extract peaks in preset frequency bands and return a pandas dataframe"""

        df_list = []
        for ix, t in enumerate(times):
            #TODO: this is super inefficient -> improve
            cur_peak_df = pd.DataFrame({cur_band : get_band_peak_fg(fg, bands[cur_band])[ix] for cur_band in bands.labels})
            cur_peak_df['peak_params'] = ['cf', 'pw', 'bw']
            cur_peak_df['time'] = t
            cur_peak_df['n_peaks'] = fg.n_peaks_[ix]
            cur_peak_df['r_squared'] = fg.get_params('r_squared')[ix]
            cur_peak_df['error'] = fg.get_params('error')[ix]
            df_list.append(cur_peak_df)
        
        df_peaks = pd.concat(df_list)
        return df_peaks



    def _get_aperiodic(self, 
                       fg, times, aperiodic_mode):

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
        df_ap['time'] = times
        
        return df_ap