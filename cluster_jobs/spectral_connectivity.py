#%%
from pathlib import Path
import joblib
import pandas as pd
import pymc as pm
#from pymc.sampling_jax import sample_numpyro_nuts
import numpy as np
import bambi as bmb

from plus_slurm import Job
import arviz as az

import sys
sys.path.append('/mnt/obob/staff/fschmidt/resting_tinnitus/utils')

from gcmi import gcmi_cc

from plus_slurm import Job

# %%
class SpecCon(Job):

    def run(self,
            low_freq=1,
            up_freq=98,
            periodic_type = None,
            feature = "exponent",
            ):

            # %%
            # low_freq=1
            # up_freq=98
            # periodic_type = 'cf'
            # feature = "alpha"


            all_files = list(Path('/mnt/obob/staff/fschmidt/resting_tinnitus/data/specparam_temporal_4s').glob(f'*/*[[]{low_freq}, {up_freq}[]].dat'))

            periodic_list, aperiodic_list = [], []

            for f in all_files:
                
                cur_data = joblib.load(f)

                periodic_list.append(cur_data['periodic'])
                aperiodic_list.append(cur_data['aperiodic'])

            physio = ['ECG', 'EOGV', 'EOGH']
            df_aperiodic = pd.concat(aperiodic_list).query('ch_name != @physio')
            df_periodic = pd.concat(periodic_list).query('ch_name != @physio')

            if np.logical_and(periodic_type != None, feature in ['exponent', 'offset', 'knee_freq', 'n_peaks']):
                return print('this doesnt make sense')
            elif np.logical_and(periodic_type == None, feature in ['delta', 'theta', 'alpha', 'beta', 'gamma']):
                return print('this also makes no sense')
        
            else:

                if periodic_type != None:
                    cur_df = df_periodic.query(f'peak_params == "{periodic_type}"')
                else:
                    cur_df = df_aperiodic

                #%%
                knee_settings = joblib.load('/mnt/obob/staff/fschmidt/resting_tinnitus/data/knee_settings.dat')
                knee_chans = knee_settings['knee']
                fixed_chans = knee_settings['fixed']

                cur_df = pd.concat([cur_df.query("ch_name == @knee_chans").query('aperiodic_mode == "knee"'),
                                    cur_df.query("ch_name == @fixed_chans").query('aperiodic_mode == "fixed"')])


                df_tin = cur_df.query('tinnitus == True')
                df_no_tin = cur_df.query('tinnitus == False')

                cols_of_interest = [feature, 'subject_id', 'ch_name', 'time']

                tin_corr = (df_tin[cols_of_interest]
                                    .pivot_table(index=['subject_id', 'time'], columns='ch_name', values=feature)
                                    .groupby('subject_id')
                                    .corr()#gcmi_cc)
                                    )

                no_tin_corr = (df_no_tin[cols_of_interest]
                                    .pivot_table(index=['subject_id', 'time'], columns='ch_name', values=feature)
                                    .groupby('subject_id')
                                    .corr()#gcmi_cc)
                                    )


                data = {'tinnitus': tin_corr,
                        'no_tinnitus': no_tin_corr}
                
                #%% stack and clean
                #tinnitus
                tin_stack = (np.abs(tin_corr).stack()
                                .rename_axis(('subject_id', 'ch_a', 'ch_b'))
                                .reset_index(name='connectivity'))

                mask_t = (tin_stack[['ch_a', 'ch_b']].apply(frozenset, axis=1).duplicated()) | (tin_stack['ch_a']==tin_stack['ch_b'])
                tin_stack = tin_stack[~mask_t]
                tin_stack['tinnitus'] = True
                
                #no tinnitus
                no_tin_stack = (np.abs(no_tin_corr).stack()
                                    .rename_axis(('subject_id', 'ch_a', 'ch_b'))
                                    .reset_index(name='connectivity'))

                mask_nt = (no_tin_stack[['ch_a', 'ch_b']].apply(frozenset, axis=1).duplicated()) | (no_tin_stack['ch_a']==no_tin_stack['ch_b'])
                no_tin_stack = no_tin_stack[~mask_nt]
                no_tin_stack['tinnitus'] = False

            
                df_stack = pd.concat([tin_stack, no_tin_stack])

                #%%
                df_stack['ch_cmb'] = df_stack['ch_a'].to_numpy() + '___' + df_stack['ch_b'].to_numpy()

                #%% run inference                
                md = bmb.Model(data=df_stack, formula='connectivity ~ 1 + tinnitus + (1 + tinnitus|ch_cmb)', dropna=True)
                
                sample_kwargs = {#'progressbar':False,
                            'draws': 2000,
                            'tune': 2000,
                            'chains': 4,
                            'target_accept': 0.95,}


                mdf = md.fit(**sample_kwargs)
                summary = az.summary(mdf, hdi_prob=.89)

                data['summary'] = summary
                data['stacked'] = df_stack

                #%%
                if periodic_type != None:
                    outpath = f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/connectivity/{feature}_{periodic_type}.dat'
                else:
                    outpath = f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/connectivity/{feature}.dat'
                
                joblib.dump(data, outpath)
