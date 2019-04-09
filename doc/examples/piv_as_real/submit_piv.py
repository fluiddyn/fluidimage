#! /usr/bin/env python
"""
Script to submit PIV jobs on a cluster (submit_piv.py)
======================================================

To be launched in bash::

  ./submit_piv.py

"""

from fluiddyn.clusters.legi import Calcul8 as Calcul

cluster = Calcul()

# we use half of the node
nb_cores = cluster.nb_cores_per_node//2

for iexp in range(4):
    command = f'job_piv.py {iexp} {nb_cores}'

    cluster.submit_script(
        command, name_run=f'fluidimage_exp{iexp}',
        nb_cores_per_node=nb_cores,
        walltime='4:00:00',
        omp_num_threads=1, idempotent=True, delay_signal_walltime=600,
        ask=False)
