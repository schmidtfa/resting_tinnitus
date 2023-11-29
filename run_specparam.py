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
#subject_ids = df_all['subject_id'].unique()

subject_ids = ['19930120laat', '19910703eigl']
#% put in jobs...
job_cluster.add_job(Specparam,
                    subject_id=PermuteArgument(subject_ids),
                    aperiodic_mode=PermuteArgument(['knee', 'fixed']),
                    peak_threshold=2,
                    #min_peak_height=0.1, #as in donoghue paper
                    freq_range=(0.25, 98),
                    )
#% submit...
job_cluster.submit(do_submit=True)

# %%
