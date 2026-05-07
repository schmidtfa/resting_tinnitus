#%% imports
from cluster_jobs.age_tinnitus_meg_osc_logit import AgeReg
#from cluster_jobs.age_tinnitus_meg_osc_t import AgeReg
from plus_slurm import JobCluster, PermuteArgument

#feature_list = [#'Offset', 
                
                #'Exponent_2', 
                #'Exponent_2', 
                #'Knee Frequency (Hz)', 
                #'tau', 
                #'alpha_cf'
 #               ]

# %%

job_cluster = JobCluster(required_ram='10G',
                        request_time=60*12,
                        request_cpus=4,
                        #qos='high_prio',
                        python_bin='/home/schmidtfa/experiments/resting_tinnitus/.pixi/envs/default/bin/python')


#% put in jobs...
job_cluster.add_job(AgeReg,
                    thresh=PermuteArgument([2.0, 3.0]),
                    feature='alpha_osc'
                                             )#feature,#
#% submit...
job_cluster.submit(do_submit=True)
# %%
