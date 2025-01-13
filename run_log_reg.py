#%% imports
from cluster_jobs.log_reg import LogReg
from cluster_jobs.lpm import LPM
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd

#% get jobcluster
job_cluster = JobCluster(required_ram='5G',
                         request_time=60*8,
                         request_cpus=4,
                         #qos='high_prio',
                         python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')

feature_list = ['delta_cf','theta_cf', 'alpha_cf', 'beta_cf', #'gamma_cf',
                'delta_pw', 'theta_pw', 'alpha_pw', 'beta_pw', #'gamma_pw',
                'delta_bw', 'theta_bw', 'alpha_bw', 'beta_bw',
                'delta_osc', 'theta_osc', 'alpha_osc', 'beta_osc',
                'Exponent_1', 'Exponent_2', #'Knee Frequency (Hz)', 
                'Offset', 'tau', 'n_peaks']

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
subject_list = list(df_all['subject_id'].unique())

#% put in jobs...
job_cluster.add_job(LogReg,
                    feature=PermuteArgument(feature_list),
                    model_type=PermuteArgument(['hi', 'up',])#'gp',])
                    )
#% submit...
job_cluster.submit(do_submit=True)

#job_cluster.run_local()
# %%

