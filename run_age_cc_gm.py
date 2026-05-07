#%% imports
from cluster_jobs.age_camcan_mri import AgeReg

from plus_slurm import JobCluster, PermuteArgument


cortex = [True,
          #False
          ]

        
job_cluster = JobCluster(required_ram='10G',
                        request_time=60*24*4, #60*12,
                        request_cpus=4,
                        extra_slurm_args=['--exclude=node01.scc-pilot.plus.ac.at,node03.scc-pilot.plus.ac.at,node04.scc-pilot.plus.ac.at'],
 
                        #qos='high_prio',
                        python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')

#% put in jobs...
job_cluster.add_job(AgeReg,
                    cortex=PermuteArgument(cortex),
                    )
#% submit...
job_cluster.submit(do_submit=True)

# %%
