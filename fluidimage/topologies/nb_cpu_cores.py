import os
import re
from multiprocessing import cpu_count
from warnings import warn

from fluidimage.config import get_config

config = get_config()

if "OMP_NUM_THREADS" not in os.environ:
    warn("OMP_NUM_THREADS not set")
else:
    OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
    if OMP_NUM_THREADS > 1:
        warn("OMP_NUM_THREADS is greater than 1!")

nb_cores = cpu_count()

allow_hyperthreading = False

if config is not None:
    try:
        allow_hyperthreading = eval(config["topology"]["allow_hyperthreading"])
    except KeyError:
        pass

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

    if nb_cores == siblings * 2:
        if not allow_hyperthreading:
            print("We do not use hyperthreading.")
            nb_cores //= 2

except IOError:
    pass
