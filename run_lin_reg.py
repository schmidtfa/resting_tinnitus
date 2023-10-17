#%% imports
from cluster_jobs.lin_reg import LinReg
from plus_slurm import ApptainerJobCluster, PermuteArgument
import pandas as pd

#% get jobcluster
job_cluster = ApptainerJobCluster(required_ram='4G',
                         request_time=1000,
                         request_cpus=4,
                         apptainer_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop_minimal_bullseye:latest',
                         python_bin='/mnt/obob/staff/fschmidt/resting_tinnitus/.venv/bin/python')

feature_list = [ 'theta', 'alpha', 'delta', 'knee_freq', 'beta', 'exponent', 'offset', 'gamma', 'n_peaks']
feature_list = ['exponent'],#'gamma', 'offset',]

df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = list(df_all['subject_id'].unique())

#% put in jobs...
job_cluster.add_job(LinReg,
                    subject_list = subject_ids,
                    feature=PermuteArgument(feature_list),
                    low_freq=0.25,
                    up_freq=98,
                    periodic_type=PermuteArgument(['cf', 'pw', None]),
                    )
#%% submit...
job_cluster.submit(do_submit=True)

#job_cluster.run_local()
# %%
