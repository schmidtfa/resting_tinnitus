#%% imports
from cluster_jobs.bart_tinnitus import BART
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd

#% get jobcluster
job_cluster = JobCluster(required_ram='10G',
                         request_time=1000,
                         request_cpus=4,
                         exclude_nodes='scs1-7,scs1-8,scs1-9,scs1-10,scs1-12,scs1-13,scs1-14',
                         #apptainer_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop_minimal_bullseye:latest',
                         python_bin='/mnt/obob/staff/fschmidt/resting_tinnitus/.venv/bin/python')


df_cmb = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_all_spec_features.csv')
chs = list(df_cmb['ch_name'].unique())

tree_list = [50, 100, 150, 200]

import numpy as np
from os import listdir

n_trees = 100
OUTDIR = f'/mnt/obob/staff/fschmidt/resting_tinnitus/data/bart/n_trees_{n_trees}'
outfiles = listdir(OUTDIR)

fname_list = []

for ch in chs:
    for fold in np.arange(5):
        fname_list.append(f'{ch}_fold_{fold}.nc')


final_files = list(set(fname_list) - set(outfiles))


c_chs = []
for f in final_files:
    c_chs.append('_'.join(f.split('_')[:-2]))

chs = list(set(c_chs))

#%% check if data already exists

#% put in jobs...
job_cluster.add_job(BART,
                    cur_ch=PermuteArgument(chs),
                    
                    n_trees=n_trees
                    )
#%% submit...
job_cluster.submit(do_submit=True)

#job_cluster.run_local()
# %%
