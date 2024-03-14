"""Parse and analyze logging files (:mod:`fluidimage.topologies.log`)
=====================================================================

.. autoclass:: LogTopology
   :members:
   :private-members:

"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fluiddyn.util import is_run_from_ipython

if is_run_from_ipython():
    plt.ion()


def float_no_valueerror(word):
    try:
        return float(word)
    except ValueError:
        return np.nan


class DataLogFile:

    def __init__(self, path_file):

        self.works = works = []
        self.works_ended = works_ended = []
        self.nb_cpus_allowed = None
        self.nb_max_workers = None
        self.executor_name = None
        self.topology_name = None
        self._title = str(path_file)

        print(f"Parsing log file: {path_file}")
        with open(path_file, "r", encoding="utf-8") as logfile:
            for iline, line in enumerate(logfile):
                if iline % 100 == 0:
                    print(f"\rparse line {iline}", end="", flush=True)

                if line.startswith("ERROR: "):
                    continue

                if self.nb_cpus_allowed is None and line.startswith(
                    "  nb_cpus_allowed = "
                ):
                    self.nb_cpus_allowed = int(line.split()[2])
                    self._title += f", nb_cpus_allowed = {self.nb_cpus_allowed}"

                if self.nb_max_workers is None and line.startswith(
                    "  nb_max_workers = "
                ):
                    self.nb_max_workers = int(line.split()[2])
                    self._title += f", nb_max_workers = {self.nb_max_workers}"

                if self.topology_name is None:
                    begin = "  topology: "
                    if line.startswith(begin):
                        self.topology_name = line.split(begin)[1].strip()

                if self.executor_name is None:
                    begin = "  executor: "
                    if line.startswith(begin):
                        self.executor_name = line.split(begin)[1].strip()

                if ". mem usage: " in line:
                    # to remove the characters coding for color
                    line = line[5:]
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

                    if ": starting execution. mem usage" in line:
                        self.date_start = date
                        self.mem_start = mem
                        time_start = t
                    elif ": end of `compute`. mem usage" in line:
                        self.date_end = date
                        self.duration = t - time_start
                        self.mem_end = mem

                if line.startswith("work ") and " done in " in line:
                    words = line.split()
                    name = words[1]
                    key = words[2][1:-1]
                    try:
                        duration = float(words[-2])
                    except ValueError:
                        pass
                    else:
                        works_ended.append(
                            {"name": name, "key": key, "duration": duration}
                        )

        print("\rdone" + 20 * " ")


class LogTopology:
    """Parse and analyze logging files."""

    def __init__(self, path):

        # path can point towards:
        # - the result directory
        # - a log file
        # - a log directory

        path = Path(path)

        if path.is_dir() and not path.name.startswith("log_"):
            paths = sorted(path.glob("log_*"))
            if not paths:
                raise ValueError("No log files found in the current directory.")
            # last saved file
            path = paths[-1]

        if path.is_file():
            path_log_file = path
            path_log_dir = path.parent
        else:
            path_log_dir = path
            paths = sorted(path_log_dir.glob("log_*.txt"))
            if not paths:
                raise ValueError(f"No log files found in {path_log_dir}.")
            path_log_file = paths[-1]

        self.path_log_file = path_log_file
        self.path_log_dir = path_log_dir
        self._title = str(path_log_file.name)

        data_main_file = DataLogFile(path_log_file)

        self.works = data_main_file.works
        self.works_ended = data_main_file.works_ended
        self.nb_cpus_allowed = data_main_file.nb_cpus_allowed
        self.nb_max_workers = data_main_file.nb_max_workers
        self.executor_name = data_main_file.executor_name
        self.topology_name = data_main_file.topology_name

        self.mem_start = data_main_file.mem_start
        self.mem_end = data_main_file.mem_end

        self.paths_log_files = sorted(path_log_dir.glob("process_*.txt"))
        for path_log_process in self.paths_log_files:
            data = DataLogFile(path_log_process)
            self.works.extend(data.works)
            self.works_ended.extend(data.works_ended)

        self.names_works = names_works = []
        for work in self.works:
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

            works_ended_name = [
                work for work in self.works_ended if work["name"] == name
            ]
            index_vs_keys = {
                work["key"]: index for index, work in enumerate(works_ended_name)
            }

            durations[name] = []
            for key in keys[name]:
                try:
                    index_key = index_vs_keys[key]
                except KeyError:
                    founded = False
                else:
                    founded = True
                    work = works_ended_name[index_key]
                    durations[name].append(work["duration"])

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
            (l,) = ax.plot(times, durations, f"C{i}o")
            lines.append(l)

            for it, t in enumerate(times):
                d = durations[it]
                ax.plot([t, t + d], [d, d], f"C{i}")

            d = np.nanmean(durations)
            ax.plot([times.min(), times.max()], [d, d], f"C{i}:", linewidth=2)

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
            (l,) = ax.plot(times[name], nb_workers[name], f"C{i}-")
            lines.append(l)

        ax.legend(lines, names, loc="center left", fontsize="x-small")

        plt.show()
