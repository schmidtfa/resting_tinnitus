#%% imports
from cluster_jobs.log_reg import LogReg
from plus_slurm import ApptainerJobCluster, PermuteArgument
import pandas as pd

#% get jobcluster
job_cluster = ApptainerJobCluster(required_ram='20G',
                         request_time=10_000,
                         request_cpus=4,
                         exclude_nodes='scs1-7,scs1-8,scs1-9,scs1-10,scs1-12,scs1-13,scs1-14',
                         apptainer_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop_minimal_bullseye:latest',
                         python_bin='/mnt/obob/staff/fschmidt/resting_tinnitus/.venv/bin/python')

feature_list = ['theta_cf', 'alpha_cf', 'beta_cf', 'gamma_cf',
                'theta_pw', 'alpha_pw', 'beta_pw', 'gamma_pw', 
                'exponent', 'offset', 'n_peaks']

df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
subject_list = list(df_all['subject_id'].unique())

#% put in jobs...
job_cluster.add_job(LogReg,
                    feature=PermuteArgument(feature_list),
                    )
#%% submit...
job_cluster.submit(do_submit=True)

#job_cluster.run_local()
# %%

