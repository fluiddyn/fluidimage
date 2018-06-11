"""Parse and analyze logging files (:mod:`fluidimage.topologies.log`)
=====================================================================

.. autoclass:: LogTopology
   :members:
   :private-members:

"""
from __future__ import print_function

from glob import glob
import os
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

from fluiddyn.util import is_run_from_ipython

if is_run_from_ipython():
    plt.ion()

colors = ["r", "b", "y", "g"]


class LogTopology(object):
    """Parse and analyze logging files.

    """

    def __init__(self, path):

        if os.path.isdir(path):
            pattern = os.path.join(path, "log_*.txt")
            paths = glob(pattern)
            paths.sort()
            if len(paths) == 0:
                raise ValueError("No log files found in the current directory.")

            path = paths[-1]

        self.log_file = os.path.basename(path)
        self._title = self.log_file

        self._parse_log(path)

    def _parse_log(self, path):
        self.works = works = []
        self.works_ended = works_ended = []
        nb_cpus = None
        nb_max_workers = None
        with open(path, "r") as f:
            print("Parsing log file: ", path)
            for iline, line in enumerate(f):
                if iline % 100 == 0:
                    print("\rparse line {}".format(iline), end="")
                    sys.stdout.flush()

                if line.startswith("ERROR: "):
                    continue

                if nb_cpus is None and line.startswith("nb_cpus_allowed = "):
                    self.nb_cpus_allowed = nb_cpus = int(line.split()[2])
                    self._title += ", nb_cpus_allowed = {}".format(nb_cpus)

                if nb_max_workers is None and line.startswith(
                    "nb_max_workers = "
                ):
                    self.nb_max_workers = nb_max_workers = int(line.split()[2])
                    self._title += ", nb_max_workers = {}".format(nb_max_workers)

                if line.startswith("INFO: ") and ". mem usage: " in line:
                    line = line[11:]
                    words = line.split()

                    try:
                        mem = float(words[-2])
                    except ValueError:
                        pass

                    if ". Launch work " in line:
                        name = words[4]
                        key = words[5][1:-2]
                        t = float(words[0])
                        works.append(
                            {
                                "name": name,
                                "key": key,
                                "mem_start": mem,
                                "time": t,
                            }
                        )
                    else:
                        date = words[0][:-1]
                        t = time.mktime(
                            time.strptime(date[:-3], "%Y-%m-%d_%H-%M-%S")
                        ) + float(date[-3:])

                    if "start compute. mem usage:" in line:
                        self.date_start = date
                        self.mem_start = mem
                        time_start = t
                    elif ": end of `compute`. mem usage" in line:
                        self.date_end = date
                        self.duration = t - time_start
                        self.mem_end = mem

                if line.startswith("INFO: work "):
                    words = line.split()
                    name = words[2]
                    key = words[3][1:-1]
                    duration = float(words[-2])
                    works_ended.append(
                        {"name": name, "key": key, "duration": duration}
                    )

                self.names_works = names_works = []
                for work in works:
                    if work["name"] not in names_works:
                        names_works.append(work["name"])

        self.durations = durations = {}
        self.times = times = {}
        self.keys = keys = {}
        for name in self.names_works:
            times[name] = []
            keys[name] = []
            for work in self.works:
                if work["name"] == name:
                    times[name].append(work["time"])
                    keys[name].append(work["key"])

            durations[name] = []
            for key in keys[name]:
                founded = False
                for work in self.works_ended:
                    if work["name"] == name and work["key"] == key:
                        durations[name].append(work["duration"])
                        founded = True
                        break

                if not founded:
                    durations[name].append(np.nan)

    def plot_memory(self):
        """Plot the memory usage versus time."""
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("memory (Mo)")
        ax.set_title(self._title, fontdict={"fontsize": 12})

        memories = np.empty(len(self.works))
        times = np.empty(len(self.works))
        for i, work in enumerate(self.works):
            memories[i] = work["mem_start"]
            times[i] = work["time"]

        ax.plot(times, memories, "o-")
        ax.plot(0, self.mem_start, "x")
        if hasattr(self, "duration"):
            ax.plot(self.duration, self.mem_end, "x")
        plt.show()

    def plot_durations(self):
        """Plot the duration of the works."""
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("duration (s)")
        ax.set_title(self._title, fontdict={"fontsize": 12})

        lines = []

        for i, name in enumerate(self.names_works):
            times = np.array(self.times[name])
            durations = self.durations[name]
            l, = ax.plot(times, durations, colors[i] + "o")
            lines.append(l)

            for it, t in enumerate(times):
                d = durations[it]
                ax.plot([t, t + d], [d, d], colors[i])

            d = np.nanmean(durations)
            ax.plot(
                [times.min(), times.max()], [d, d], colors[i] + "-", linewidth=2
            )

        ax.legend(lines, self.names_works, loc="center left", fontsize="x-small")

        plt.show()

    def plot_nb_workers(self, str_names=None):
        """Plot the number of workers versus time."""
        if str_names is not None:
            names = [name for name in self.names_works if str_names in name]
        else:
            names = self.names_works

        nb_workers = {}
        times = {}
        for name in names:
            times_start = list(self.times[name])
            times_stop = list(
                np.array(times_start) + np.array(self.durations[name])
            )

            deltas = np.array([1 for t in times_start] + [-1 for t in times_stop])
            times_unsorted = np.array(times_start + times_stop)

            argsort = np.argsort(times_unsorted)

            ts = times_unsorted[argsort]
            nbws = np.cumsum(deltas[argsort])

            ts2 = []
            nbws2 = []
            for i, t in enumerate(ts[:-1]):
                nbw = nbws[i]
                ts2.append(t)
                ts2.append(ts[i + 1])
                nbws2.append(nbw)
                nbws2.append(nbw)

            times[name] = ts2
            nb_workers[name] = nbws2

        plt.figure()
        ax = plt.gca()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("number of workers")
        ax.set_title(self._title, fontdict={"fontsize": 12})

        lines = []

        for i, name in enumerate(self.names_works):
            l, = ax.plot(times[name], nb_workers[name], colors[i] + "-")
            lines.append(l)

        ax.legend(lines, names, loc="center left", fontsize="x-small")

        plt.show()
