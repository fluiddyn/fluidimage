from warnings import warn
import re
import os
from multiprocessing import cpu_count

from fluidimage.config import get_config

config = get_config()

if "OMP_NUM_THREADS" not in os.environ:
    warn("OMP_NUM_THREADS not set")
else:
    OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
    if OMP_NUM_THREADS > 1:
        warn("OMP_NUM_THREADS is greater than 1!")

nb_cores = cpu_count()

if config is not None:
    try:
        allow_hyperthreading = eval(config["topology"]["allow_hyperthreading"])
    except KeyError:
        allow_hyperthreading = True

try:  # should work on UNIX
    # found in http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python # noqa
    with open("/proc/self/status") as f:
        status = f.read()
    m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", status)
    if m:
        nb_cpus_allowed = bin(int(m.group(1).replace(",", ""), 16)).count("1")

    if nb_cpus_allowed > 0:
        nb_cores = nb_cpus_allowed

    with open("/proc/cpuinfo") as f:
        cpuinfo = f.read()

    nb_proc_tot = 0
    siblings = None
    for line in cpuinfo.split("\n"):
        if line.startswith("processor"):
            nb_proc_tot += 1
        if line.startswith("siblings") and siblings is None:
            siblings = int(line.split()[-1])

    if nb_proc_tot == siblings * 2:
        if allow_hyperthreading is False:
            print("We do not use hyperthreading.")
            nb_cores //= 2

except IOError:
    pass

nb_max_workers = None
if config is not None:
    try:
        nb_max_workers = eval(config["topology"]["nb_max_workers"])
    except KeyError:
        pass

# default nb_max_workers
# Difficult: trade off between overloading and limitation due to input output.
# The user can do much better for a specific case.
if nb_max_workers is None:
    if nb_cores < 16:
        nb_max_workers = nb_cores + 2
    else:
        nb_max_workers = nb_cores
