#%% imports
from cluster_jobs.aperiodic_analysis import rasa
from cluster_jobs.periodic_analysis import peak_rasa
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd#

#%% Run aperiodic part
job_cluster = JobCluster(required_ram='10G',
                         request_time=60*1,
                         request_cpus=2,
                         qos='high_prio',
                         python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

#% put in jobs...
job_cluster.add_job(rasa,
                    subject_id=PermuteArgument(subject_ids),
                    cur_ic=PermuteArgument(['AIC', 'BIC'])
                    )
#%% submit...
job_cluster.submit(do_submit=True)

#%% run periodic part
#% get jobcluster

df_all = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

job_cluster = JobCluster(required_ram='5G',
                         request_time=60,
                         request_cpus=1,
                         qos='high_prio',
                         python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')
#% put in jobs...
job_cluster.add_job(peak_rasa,
                    subject_id=PermuteArgument(subject_ids),
                    peak_threshold=PermuteArgument([2., 3.])
                    #min_peak_height=0.01,
                    )
#% submit...
job_cluster.submit(do_submit=True)# %%

# %%
