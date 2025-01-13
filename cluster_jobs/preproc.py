#%%
from cluster_jobs.meta_job import Job
#%%
import joblib
import pandas as pd
import scipy.signal as dsp
from neurodsp.spectral import compute_spectrum_welch
from neurodsp.aperiodic.irasa import compute_irasa

from pyrasa import irasa

import sys
sys.path.append('/home/schmidtfa/experiments/resting_tinnitus/utils')
from preproc_utils import preproc_data
from src_utils import raw2source
import mne
import numpy as np
from os.path import join

#%%

class PreprocessingJob(Job):

    job_data_folder = 'data_meg'

    def run(self,
            subject_id,
            max_filt=True,
            l_pass = None,
            h_pass = 0.1,
            notch = False,
            do_ica = True,
            src_type='beamformer',
            source='volume',
            downsample_f = 1000, #make sure that the 10 or 5k data is also at 1k
            ica_threshold = 0.5,
            hmax=2.,
            duration=4,
            atlas='destrieux'):


        #%% debug
        # subject_id = '19891222gbhl'
        # l_pass = 98
        # h_pass = 1
        # notch = False
        # do_ica = False
        # ica_threshold = 0.5
        # max_filt=True
        # downsample_f = None
        # duration=4
        # src_type='beamformer'
        # source='volume'
        # atlas='dk'
        # hmax=2.

        if atlas == 'dk':
             vol_atlas = 'aparc+aseg'
             surf_atlas = 'aparc'
        elif atlas == 'destrieux':
             vol_atlas = 'aparc.a2009s+aseg'
             surf_atlas = 'aparc.a2009s'
        elif atlas == 'glasser':
             if source == 'volume':
                ValueError('No volumetric model for the glasser atlas available')
             surf_atlas = 'HCPMMP1'

        df = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv').query(f'subject_id == "{subject_id}"')
        cur_path = join('/home/schmidtfa/experiments/resting_tinnitus/data/sinuhe/', df['path'].iloc[0].split('/')[-1])

        preproc_settings = {'max_filt': max_filt, 
                            'notch': notch,
                            'coord_frame':'head',
                            'l_pass': l_pass, 
                            'h_pass': h_pass, 
                            'do_ica': do_ica, 
                            'ica_threshold': ica_threshold,
                            'downsample_f': downsample_f}

        raw = preproc_data(cur_path, **preproc_settings)
        
        #% crop data (if it exceeds 5min -> typical this is not the case only for some p10 subjects we did 15min recordings)
        if raw.times.max() / 60 > 5:
                raw.crop(0, 5*60)

        eog = raw.get_data(picks='eog')
        ecg = raw.get_data(picks='ecg')

        fs = raw.info['sfreq']
        welch_settings = {'fs': fs, 
                        'avg_type':'median',
                        'window': 'hann',
                        'nperseg': fs*duration,
                        'noverlap': fs*duration / 2, #50% overlap as default
                        'outlier_percent': 5}

        #%% move data to source
        #adjust preprocessing settings for empty room
        preproc_settings['coord_frame'] = 'meg'
        preproc_settings['do_ica'] = False
        subjects_dir = '/home/schmidtfa/experiments/resting_tinnitus/data/freesurfer'

        stc = raw2source(raw, subject_id, subjects_dir, preproc_settings, src_type=src_type, source=source)

        #% get tc from parcellation and return
        fs_path = join(subjects_dir, f'{subject_id}_from_template')
        if source == 'volume':
                src_file = f'{fs_path}/bem/{subject_id}_from_template-vol-10-src.fif'
                src = mne.read_source_spaces(src_file)
                labels_mne = join(fs_path, 'mri/' + vol_atlas + '.mgz')
                label_names = mne.get_volume_labels_from_aseg(labels_mne)

                ctx_logical = [True if 'ctx' in label else False for label in label_names]
                sctx_logical = [True if f == False else False for f in ctx_logical]
                
                ctx_labels = [label[4:] for label in label_names if 'ctx' in label]
                sctx_labels = list(np.array(label_names)[sctx_logical])
                rh = [True if label[:2] == 'rh' else False for label in ctx_labels]
                lh = [True if label[:2] == 'lh' else False for label in ctx_labels]

                label_info = {'lh': lh,
                              'rh': rh,
                              'parc': vol_atlas + '.mgz',
                              'ctx_labels': ctx_labels,
                              'ctx_logical': ctx_logical,
                              'sctx_logical': sctx_logical,
                              'sctx_labels': sctx_labels}
        elif source == 'surface':
                src_file = f'{fs_path}/bem/{subject_id}_from_template-ico-4-src.fif'
                src = mne.read_source_spaces(src_file)
                labels_mne = mne.read_labels_from_annot(f'{subject_id}_from_template', 
                                                        parc=surf_atlas, 
                                                        subjects_dir=subjects_dir)
                names_order_mne = np.array([label.name[:-3] for label in labels_mne])

                rh = [True if label.hemi == 'rh' else False for label in labels_mne]
                lh = [True if label.hemi == 'lh' else False for label in labels_mne]

                label_info = {'lh': lh,
                                'rh': rh,
                                'parc': surf_atlas,
                                'names_order_mne': names_order_mne}
        
        #%%
        welch_settings = { 
                'avg_type':'median',
                'nperseg': int(fs*duration),
                'noverlap': int(fs*duration / 2), #50% overlap as default
                }
        irasa_kwargs = {'fs': fs,
                        'band': (1, 100),
                        'psd_kwargs': welch_settings,
                        'hset_info': (1.05, hmax, 0.05)}

        eog_psd = irasa(eog, **irasa_kwargs)
        ecg_psd = irasa(ecg, **irasa_kwargs)

        # mean flip time series costs significantly less memory than averaging the irasa'd spectra
        label_tc = mne.extract_label_time_course(stc, labels_mne, src, mode='mean_flip')

        irasa_label_stc = irasa(label_tc, **irasa_kwargs)
        peak_kwargs = {'min_peak_height': 0.1,
                        'peak_threshold': 1}
        aperiodic_error = irasa_label_stc.get_aperiodic_error(peak_kwargs=peak_kwargs) 

        data_dict = {'irasa_label_stc': irasa_label_stc,
                        'label_info': label_info,
                        'aperiodic_error': aperiodic_error}

        #%%
        data = {'subject_info': df,
                'eog': eog_psd,
                'ecg': ecg_psd,
                'src': data_dict,
                'subject_id': subject_id,
                }

        joblib.dump(data, self.full_output_path)