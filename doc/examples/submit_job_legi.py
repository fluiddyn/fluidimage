from fluiddyn.clusters.legi import Calcul8

cluster = Calcul8()

cluster.commands_setting_env = [
    "source /etc/profile",
    "module purge",
    "source $HOME/miniconda3/etc/profile.d/conda.sh",
    "conda activate env_fluidimage",
]

cluster.submit_script(
    "piv_with_topo_complete.py",
    name_run="fluidimage",
    nb_cores_per_node=4,
    walltime="0:15:00",
    omp_num_threads=1,
    idempotent=True,
    delay_signal_walltime=300,
)
