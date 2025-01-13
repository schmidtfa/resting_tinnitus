#%%
from cluster_jobs.meta_job import Job
#%%
from pathlib import Path
import joblib
import pandas as pd

#%%

class rasa(Job):

    job_data_folder = 'pyrasa_ap_params_duration_2'

    def run(self,
            subject_id,
            cur_ic = 'BIC',
            duration = 2
            ):

        #%%debug
        # subject_id = '19981024mrfr'#
        # cur_ic = 'BIC'

        # %%
        query_string = f'__duration_{duration}__hmax_2__source_surface__atlas_glasser.dat'
        INDIR = '/home/schmidtfa/experiments/resting_tinnitus/data/data_meg'
        cur_data = joblib.load(str(list(Path(INDIR).glob(f'{subject_id}/{subject_id}' + query_string))[0]))

        #%% src based analysis
        def add_info(df, df_info):
            df['subject_id'] = df_info['subject_id'].iloc[0]
            df['tinnitus'] = df_info['tinnitus'].iloc[0]
            df['dB'] = df_info['dB'].iloc[0]
            df['age'] = df_info['measurement_age'].iloc[0]
            df['tinnitus_distress'] = df_info['tinnitus_distress'].iloc[0]

            return df

        cur_data['src']['irasa_label_stc'].ch_names = cur_data['src']['label_info']['names_order_mne']
        fixed_model = cur_data['src']['irasa_label_stc'].fit_aperiodic_model(fit_func='fixed')
        knee_model = cur_data['src']['irasa_label_stc'].fit_aperiodic_model(fit_func='knee')

        #%%
        def pick_best_model(df_list, ic='BIC'):
            df_cmb_gof = (pd.concat(df_list)[[ic, 'ch_name', 'fit_type']]
                            .pivot(columns='fit_type', 
                                values=ic,
                                index='ch_name')
                            .reset_index()
                        )
            fixed_ctx = df_cmb_gof['ch_name'][df_cmb_gof['fixed'] < df_cmb_gof['knee']].to_numpy()
            knee_ctx = df_cmb_gof['ch_name'][df_cmb_gof['fixed'] > df_cmb_gof['knee']].to_numpy()
            return fixed_ctx, knee_ctx
            
        fixed_ctx_chs, knee_ctx_chs = pick_best_model([knee_model.gof.query('ch_name != "???"'), fixed_model.gof.query('ch_name != "???"')], ic=cur_ic)

        #%%
        #TODO: ADD ECG and EOG analysis
        data = {'ctx': {'ch_coice': {'knee':  knee_ctx_chs,
                                    'fixed': fixed_ctx_chs},
                        'aperiodic_params': {'knee': add_info(knee_model.aperiodic_params, cur_data['subject_info']),
                                            'fixed': add_info(fixed_model.aperiodic_params, cur_data['subject_info'])},
                        'models': {'knee':add_info(knee_model.model, cur_data['subject_info']),
                                'fixed': add_info(fixed_model.model, cur_data['subject_info'])}},
                'subject_info': cur_data['subject_info'],
                'ave_spec': {'psd_ave': cur_data['src']['irasa_label_stc'].raw_spectrum.mean(axis=0),
                             'ap_m_ave_k': knee_model.model.groupby('Frequency (Hz)')['aperiodic_model'].mean().to_numpy(),
                             'ap_m_ave_f': fixed_model.model.groupby('Frequency (Hz)')['aperiodic_model'].mean().to_numpy(),
                             'psd_ap': cur_data['src']['irasa_label_stc'].aperiodic.mean(axis=0),
                             'freqs': cur_data['src']['irasa_label_stc'].freqs}}
        
        joblib.dump(data, self.full_output_path)