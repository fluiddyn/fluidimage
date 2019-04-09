#! /usr/bin/env python
"""
Script to submit preprocessing jobs on a cluster (submit_pre.py)
================================================================

To be launched in bash::

  ./submit_pre.py

"""
from fluiddyn.clusters.legi import Calcul7 as Calcul

cluster = Calcul()
nb_cores = cluster.nb_cores_per_node//2


for iexp in range(4):
    # We restrict the numbers of workers for the topology to limit memory usage
    command = 'job_pre.py {} {}'.format(iexp, nb_cores//1.5)
    cluster.submit_script(
        command, name_run=f'fluidimage_preproc_exp{iexp}',
        nb_cores_per_node=nb_cores,
        walltime='4:00:00',
        omp_num_threads=1, idempotent=True, delay_signal_walltime=600,
        ask=False)
