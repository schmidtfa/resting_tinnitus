#%% imports
from cluster_jobs.rf_tinnitus import RandomForest
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd

#% get jobcluster
job_cluster = JobCluster(required_ram='1G',
                         request_time=60*4,
                         request_cpus=2,
                         qos='high_prio',
                         python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')


df_ap = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/aperiodic_params.csv')
chs = list(df_ap['ch_name'].unique())

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
