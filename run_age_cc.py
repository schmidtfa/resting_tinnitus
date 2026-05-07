#%% imports
from cluster_jobs.age_camcan_meg import AgeReg

from plus_slurm import JobCluster, PermuteArgument
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import pandas as pd
from os.path import join, exists


feature_list = [#'Exponent_1', 
                #'Exponent_2', 
                #'tau', 
                'alpha_cf',
                'alpha_pw',
                'alpha_bw'
                ]

        
job_cluster = JobCluster(required_ram='20G',
                        request_time=60*12*3, #60*12,
                        request_cpus=4,
                        #qos='high_prio',
                        python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')

#% put in jobs...
job_cluster.add_job(AgeReg,
                    feature=PermuteArgument(feature_list),
                    )
#% submit...
job_cluster.submit(do_submit=True)

# %%
