#%% imports
from cluster_jobs.specparam import Specparam
from plus_slurm import ApptainerJobCluster, PermuteArgument
import pandas as pd
#% get jobcluster
job_cluster = ApptainerJobCluster(required_ram='5G',
                         request_time=600,
                         request_cpus=2,
                         apptainer_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop_minimal_bullseye:latest',
                         python_bin='/mnt/obob/staff/fschmidt/resting_tinnitus/.venv/bin/python')



df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')
subject_ids = df_all['subject_id'].unique()

#% put in jobs...
job_cluster.add_job(Specparam,
                    subject_id=PermuteArgument(subject_ids),
                    aperiodic_mode=PermuteArgument(['knee', 'fixed']),
                    freq_range=(1, 100),
                    )
#% submit...
job_cluster.submit(do_submit=True)

# %%
