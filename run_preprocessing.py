#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:33:47 2022

@author: schmidtfa
"""
#%% imports
from cluster_jobs.preprocess_meg import PreprocessingJob
from plus_slurm import JobCluster, PermuteArgument
#%%
import pandas as pd
#%% get jobcluster
job_cluster = JobCluster(required_ram='4G',
                         request_time=400,
                         request_cpus=2,
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

#%%
tinnitus_path = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/subject_lists/df_tinnitus.csv'

df_all = pd.read_csv(tinnitus_path).query('fs_1k == True')
df_all.reset_index(inplace=True)
subject_ids = df_all['subject_id'].unique()



#%% put in jobs...
job_cluster.add_job(PreprocessingJob,
                    subject_id=PermuteArgument(subject_ids),
                    #data_frame_path=tinnitus_path,
                    freq_range = (0.5, 45),
                    meg_type = 'mag',
                    eye_threshold = 0.5,
                    heart_threshold = 0.5,
                    )
#%% submit...
job_cluster.submit(do_submit=True)

# %%
