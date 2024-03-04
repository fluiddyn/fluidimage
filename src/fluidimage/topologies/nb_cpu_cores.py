"""Utility to obtain the number of cores available

"""

import re
from multiprocessing import cpu_count

from simpleeval import simple_eval

from fluidimage.config import get_config

config = get_config()

nb_cores = cpu_count()

allow_hyperthreading = False

if config is not None:
    try:
        allow_hyperthreading = simple_eval(
            config["topology"]["allow_hyperthreading"]
        )
    except KeyError:
        pass

try:  # should work on UNIX
    # found in http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python # noqa
    with open("/proc/self/status", encoding="utf-8") as file:
        status = file.read()
    m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", status)
    if m:
        nb_cpus_allowed = bin(int(m.group(1).replace(",", ""), 16)).count("1")

    if nb_cpus_allowed > 0:
        nb_cores = nb_cpus_allowed

    with open("/proc/cpuinfo", encoding="utf-8") as file:
        cpuinfo = file.read()

    nb_proc_tot = 0
    siblings = None
    for line in cpuinfo.split("\n"):
        if line.startswith("processor"):
            nb_proc_tot += 1
        if line.startswith("siblings") and siblings is None:
            siblings = int(line.split()[-1])

    if siblings is not None and nb_cores == siblings * 2:
        if not allow_hyperthreading:
            print("We do not use hyperthreading.")
            nb_cores //= 2

except IOError:
    pass
