#%% imports
from cluster_jobs.preproc import PreprocessingJob
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd
#% get jobcluster
job_cluster = JobCluster(required_ram='40G',
                         request_time=60*1,
                         request_cpus=2,
                         qos='high_prio',
                         python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')



df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

#%% put in jobs...
job_cluster.add_job(PreprocessingJob,
                    subject_id=PermuteArgument(subject_ids),
                    duration=2,
                    hmax=PermuteArgument([2, 3, 4]),
                    source = 'surface',
                    atlas = 'glasser')
                    
#% submit...
job_cluster.submit(do_submit=True)

# %%
