
from math import ceil

from fluiddyn.clusters.legi import Calcul

cluster = Calcul()

node = 'cl7n010'
nb_min_1 = 48

for nb_cores in [6]:
    nb_min = nb_min_1/nb_cores
    str_nb_cores = f'{nb_cores}'

    print(str_nb_cores, '0:{}:00'.format(int(ceil(nb_min)) + 10))

    cluster.submit_script(
        'piv_from_uvmat_xml.py ' + str_nb_cores,
        name_run='fli_' + node + '_' + str_nb_cores,
        nb_cores_per_node=nb_cores,
        walltime='0:{}:00'.format(int(ceil(nb_min)) + 10),
        omp_num_threads=1,
        idempotent=True, delay_signal_walltime=300,
        network_address=node)
