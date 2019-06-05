
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

plt.ion()

path = 'bench_cl7n010_2016-04-29'

files = glob(path+'/*out')

times = {}
times_wo_hyperthreading = {}

for fn in files:
    allow_hyperthreading = True
    with open(fn) as f:
        for line in f:
            if re.findall(r'Cpus_allowed:', line):
                nb_p = int(line.split(' ')[-1])
            if re.findall(r'ellapsed time:', line):
                t = float(line.split(' ')[-1][:-2])
            if 'hyperthreading' in line:
                allow_hyperthreading = False

        if allow_hyperthreading:
            times[nb_p] = t
        else:
            times_wo_hyperthreading[nb_p] = t

time_ref = times_wo_hyperthreading[2]

nb_proc = times.keys()
nb_proc.sort()
nb_proc = np.array(nb_proc)
times = np.array([times[n] for n in nb_proc])
nb_proc /= 2


nb_proc_woh = times_wo_hyperthreading.keys()
nb_proc_woh.sort()
nb_proc_woh = np.array(nb_proc_woh)
times_woh = np.array([times_wo_hyperthreading[n] for n in nb_proc_woh])
nb_proc_woh /= 2


plt.figure()
ax = plt.gca()

ax.plot(nb_proc, time_ref/times, 'ob')
ax.plot(nb_proc_woh, time_ref/times_woh, 'or')

ax.set_xlabel('number of cores')
ax.set_ylabel(r'speedup compared to 1 core without hyperthreading')

tmp = [0, nb_proc.max()]
ax.plot(tmp, tmp, 'k')




ylim = list(ax.get_ylim())
ylim[0] = 0.
ax.set_ylim(ylim)


plt.show()
