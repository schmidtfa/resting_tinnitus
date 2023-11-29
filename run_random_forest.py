#%% imports
from cluster_jobs.rf_tinnitus import RandomForest
#from cluster_jobs.rf_loo import RandomForest
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd

#% get jobcluster
job_cluster = JobCluster(required_ram='4G',
                         request_time=1000,
                         request_cpus=4,
                         exclude_nodes='scs1-7,scs1-8,scs1-9,scs1-10,scs1-12,scs1-13,scs1-14',
                         #apptainer_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop_minimal_bullseye:latest',
                         python_bin='/mnt/obob/staff/fschmidt/resting_tinnitus/.venv/bin/python')


df_cmb = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_all_spec_features.csv')
chs = list(df_cmb['ch_name'].unique())

#%% check if data already exists

#% put in jobs...
job_cluster.add_job(RandomForest,
                    cur_ch=PermuteArgument(chs),
                    n_splits=5,
                    n_repeats=5,
                    )
#%% submit...
job_cluster.submit(do_submit=True)
# %%
