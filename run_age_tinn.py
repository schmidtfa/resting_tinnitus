#%% imports
from cluster_jobs.age_tinnitus_meg_mosi import AgeReg
#from cluster_jobs.age_tinnitus_meg_osc_t import AgeReg
from plus_slurm import JobCluster, PermuteArgument

feature_list = ['Offset', 
                
                'Exponent_1', 
                'Exponent_2', 
                #'Knee Frequency (Hz)', 
                'tau', 
                'alpha_cf',
                'alpha_bw',
                'alpha_pw'
                ]

# %%

job_cluster = JobCluster(required_ram='10G',
                        request_time=60*4,
                        request_cpus=4,
                        #extra_slurm_args=['--exclude=node01.scc-pilot.plus.ac.at,node03.scc-pilot.plus.ac.at,node04.scc-pilot.plus.ac.at'],
 
                        qos='high_prio',
                        python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')


#% put in jobs...
job_cluster.add_job(AgeReg,
                    thresh=2.0,#PermuteArgument([2.0, 3.0]),
                    feature=PermuteArgument(feature_list
                                             )#feature,#
                    )
#% submit...
job_cluster.submit(do_submit=True)
# %%
