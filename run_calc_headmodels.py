#%%
from cluster_jobs.calc_headmodels import HeadModelJob
from plus_slurm import ApptainerJobCluster, PermuteArgument
from plus_slurm import PermuteArgument
import pandas as pd

job_cluster = ApptainerJobCluster(required_ram='4G',
                                    request_time = 180,
                                    request_cpus=2,
                                    apptainer_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop_minimal_bullseye:latest',
                                    python_bin='/mnt/obob/staff/fschmidt/resting_tinnitus/.venv/bin/python')

df_all = pd.read_csv('/mnt/obob/staff/fschmidt/resting_tinnitus/data/tinnitus_match.csv')

subject_ids = df_all['subject_id'].unique()

#%% put in jobs...
job_cluster.add_job(
                    HeadModelJob,
                    subject_id=PermuteArgument(subject_ids)
                    )

job_cluster.submit(do_submit=True)

# %%
