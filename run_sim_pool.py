#%% imports
from cluster_jobs.test_pooling_model import SimPool
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd
import numpy as np

#% get jobcluster
job_cluster = JobCluster(required_ram='5G',
                         request_time=60*4,
                         request_cpus=4,
                         #qos='high_prio',
                         python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')


df_regions_info = pd.read_csv('/home/schmidtfa/experiments/resting_tinnitus/data/regions_hcmp.csv')
df_regions_info['roi'] = [region[-1] + '_' + region[:-2] + '_ROI' for region in df_regions_info['regionName']]
df_regions_info['roi'] = df_regions_info['roi'].replace({'L_7Pl_ROI': 'L_7PL_ROI',
                                                        'R_7Pl_ROI': 'R_7PL_ROI',})

df_regions_info['cortex_info'] = df_regions_info['cortex'] + '_' + df_regions_info['LR']


feature_list = list(np.unique([cort[:-2] for cort in df_regions_info['cortex_info'].unique()]))

seeds = np.arange(100).tolist()

#%% put in jobs...
job_cluster.add_job(SimPool,
                    cortex_of_interest=PermuteArgument(feature_list),
                    cur_seed=PermuteArgument(seeds),
                    slope_effect = [0.3, 0.4, 0.5, 0.6, 0.7],
                    n_subjects = 40,
                    )
#%% submit...
job_cluster.submit(do_submit=True)

# %%
