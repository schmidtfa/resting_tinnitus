#%%
from cluster_jobs.calc_headmodels import HeadModelJob
from plus_slurm import JobCluster, PermuteArgument
from plus_slurm import PermuteArgument
import pandas as pd

job_cluster = JobCluster(required_ram='4G',
                                  request_cpus=2,
                                  request_time=60*2,
                                  qos='high_prio',
                                  python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')


df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')

subject_ids = df_all['subject_id'].unique()

#%% put in jobs...
job_cluster.add_job(
                    HeadModelJob,
                    subject_id=PermuteArgument(subject_ids)
                    )

job_cluster.submit(do_submit=True)

# %%
