
from fluiddyn.clusters.legi import Calcul7

cluster = Calcul7()

cluster.submit_script(
    'bench_piv_work.py', name_run='fluidimage',
    nb_cores_per_node=1,
    walltime='0:02:00',
    omp_num_threads=1,
    idempotent=True, delay_signal_walltime=300)
