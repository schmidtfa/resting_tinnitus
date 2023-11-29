#%%
from cluster_jobs.meta_job import Job
#%%
from pathlib import Path
import joblib
from fooof import FOOOFGroup, Bands
from fooof.analysis import get_band_peak_fg
from fooof.utils.params import compute_knee_frequency
from fooof.utils import interpolate_spectrum

import numpy as np
import numpy.matlib
import pandas as pd

import random
random.seed(42069) #make it reproducible - sort of

#%%

class Specparam(Job):

    job_data_folder = 'specparam_full_spectra'

    def run(self,
            subject_id,
            aperiodic_mode,
            freq_range=(0.5, 100),
            max_n_peaks=6,
            min_peak_height=0,
            peak_threshold=2,
            ):

        #%%debug
        # subject_id = '19990822mrae'
        # aperiodic_mode = 'fixed'
        # freq_range=(0.25, 98)
        # max_n_peaks=2
        # min_peak_height=0
        # peak_threshold=2

        INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg'
        cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__src_type_beamformer.dat'))[0])

        #define some bands
        bands = Bands({ 'delta' : [1, 3],
                        'theta' : [3, 8], 
                        'alpha' : [8, 14],
                        'beta' : [14, 30],
                        'gamma' : [30, 45],
                        'line_noise': [49, 51]})

        #%%
        periodic_list, aperiodic_list, spectra_list = [],[],[]

        for key in ['ecg', 'eog', 'src']:

            if key == 'eog':
                ch_names = ['EOGV', 'EOGH']
                #% interpolate line noise -> TODO: for some reason this causes nans in the eog and ecg data -> check if potential effects are there
                #_,  cur_data[key] = interpolate_spectrum(np.matlib.repmat(cur_data['freq'], np.shape(ch_names)[0], 1), cur_data[key], [48, 52])
                data2fooof = cur_data[key]
            elif key == 'ecg':
                ch_names = ['ECG']
                #% interpolate line noise
                #_,  cur_data[key] = interpolate_spectrum(cur_data['freq'][np.newaxis,:], cur_data[key], [48, 52])
                #NOTE: little trick to keep using fooofgroup -> only one channel is kept in later parts of the analysis
                data2fooof = np.vstack([cur_data[key], cur_data[key]]) 
            elif key == 'src':
                ch_names = cur_data['src']['label_info']['names_order_mne']
                #_,  cur_data['src']['label_tc'] = interpolate_spectrum(np.matlib.repmat(cur_data['freq'], np.shape(ch_names)[0], 1), cur_data['src']['label_tc'], [48, 52])
                data2fooof = cur_data['src']['label_tc']

            #% do some spectral parametrization
            fg = FOOOFGroup(peak_width_limits=(0.5,6), #more doesnt really make sense to me
                            max_n_peaks=max_n_peaks, #this uses my fooof version with bic -> i.e. best model combination between parsimony and complexity is picked
                            aperiodic_mode=aperiodic_mode,
                            peak_threshold=peak_threshold,
                            min_peak_height=min_peak_height,
                            verbose=False) #otherwise log-files > 40gb

            fg.fit(cur_data['freq'], data2fooof, freq_range=freq_range)

            spectra_list.append(self._extract_spectra(fg, key, ch_names))
            
            #% get band specific information
            df_periodic = self._get_band_peak_df(fg, bands, ch_names)
            #df_periodic = _get_band_peak_df(fg, bands, ch_names)
            df_periodic['aperiodic_mode'] = aperiodic_mode

            #% get aperiodic information
            df_aperiodic = self._get_aperiodic(fg, ch_names, key, aperiodic_mode)
            #df_aperiodic = _get_aperiodic(fg, ch_names, key, aperiodic_mode)
            df_aperiodic['aperiodic_mode'] = aperiodic_mode

            periodic_list.append(df_periodic)
            aperiodic_list.append(df_aperiodic)

        #%% combine all info in one df

        df_cmb_p = pd.concat(periodic_list)
        df_cmb_a = pd.concat(aperiodic_list)

        #add demographics
        demo_list = ['subject_id', 'tinnitus', 'measurement_age', 'dB','tinnitus_distress']
        sub_info = cur_data['subject_info'][demo_list].drop_duplicates('subject_id')

        data = {}
        for key, df in zip(['periodic', 'aperiodic'], [df_cmb_p, df_cmb_a]):

            df['subject_id'] = cur_data['subject_id']
            

            data.update({key: df.merge(sub_info, on='subject_id')})

        #save
        data.update({'full_spectra': spectra_list})
        data['freq'] = cur_data['freq']
        data['label_info'] = cur_data['src']['label_info']


        joblib.dump(data, self.full_output_path)


    #%% utility functions
    def _get_band_peak_df(self, fg, bands, ch_names):

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



    def _get_aperiodic(self, fg, ch_names, key, aperiodic_mode):

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



    def _extract_spectra(self, fg, key, ch_names):

        """Extract full periodic and aperiodic model fits"""
        
        peaks, aps = {}, {}

        for ix, cur_ch in enumerate(ch_names):

            aps.update({cur_ch : fg.get_fooof(ix).get_model(component='aperiodic', space='log')})
            peaks.update({cur_ch : fg.get_fooof(ix).get_model(component='peak', space='log')})

        spectra = {key: {'aperiodic': aps,
                         'periodic': peaks}}

        return spectra

# %%
