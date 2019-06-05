"""Parse and analyze logging files (:mod:`fluidimage.topologies.log`)
=====================================================================

.. autoclass:: LogTopology
   :members:
   :private-members:

"""

import time
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fluiddyn.util import is_run_from_ipython

if is_run_from_ipython():
    plt.ion()

colors = ["r", "b", "y", "g"]


def float_no_valueerror(word):
    try:
        return float(word)
    except ValueError:
        return np.nan


class LogTopology:
    """Parse and analyze logging files.

    """

    def __init__(self, path):
        path = Path(path)

        if path.is_dir():
            paths = sorted(glob(str(path / "log_*")))
            paths = [path for path in paths if "_multi" not in Path(path).name]
            if len(paths) == 0:
                raise ValueError("No log files found in the current directory.")

            path = Path(paths[-1])
            if path.is_dir():
                path = Path(
                    next(
                        path
                        for path in glob(str(path / "log*"))
                        if "_multi" not in path
                    )
                )

        self.log_dir_path = path.parent
        self.log_file = path.name
        self._title = str(self.log_file)

        self._parse_log(path)

    def _parse_log(self, path):
        self.works = works = []
        self.works_ended = works_ended = []
        self.nb_cpus_allowed = None
        self.nb_max_workers = None
        self.log_files = None
        self.executor_name = None
        self.topology_name = None
        with open(path, "r") as logfile:
            print("Parsing log file: ", path)
            for iline, line in enumerate(logfile):
                if iline % 100 == 0:
                    print(f"\rparse line {iline}", end="", flush=True)

                if line.startswith("ERROR: "):
                    continue

                if self.nb_cpus_allowed is None and line.startswith(
                    "INFO:   nb_cpus_allowed = "
                ):
                    self.nb_cpus_allowed = int(line.split()[3])
                    self._title += f", nb_cpus_allowed = {self.nb_cpus_allowed}"

                if self.nb_max_workers is None and line.startswith(
                    "INFO:   nb_max_workers = "
                ):
                    self.nb_max_workers = int(line.split()[3])
                    self._title += f", nb_max_workers = {self.nb_max_workers}"

                if self.topology_name is None:
                    begin = "INFO:   topology: "
                    if line.startswith(begin):
                        self.topology_name = line.split(begin)[1].strip()

                if self.executor_name is None:
                    begin = "INFO:   executor: "
                    if line.startswith(begin):
                        self.executor_name = line.split(begin)[1].strip()

                if self.log_files is None:
                    begin = "INFO: logging files: "
                    if line.startswith(begin):
                        self.log_files = eval(line.split(begin)[1].strip())

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

                    if ": starting execution. mem usage" in line:
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

        print("\rdone" + 20 * " ")

        if self.log_files is not None:
            path_dir = self.log_dir_path
            for file_name in self.log_files:
                path = path_dir / file_name
                with open(path, "r") as logfile:
                    print("Parsing log file: ", path.name)
                    for iline, line in enumerate(logfile):
                        if iline % 100 == 0:
                            print(f"\rparse line {iline}", end="", flush=True)

                        if line.startswith("ERROR: "):
                            continue

                        if line.startswith("INFO: ") and ". mem usage: " in line:
                            line = line[11:]
                            words = line.split()
                            mem = float_no_valueerror(words[-2])

                            if ". Launch work " in line:
                                name = words[4]
                                key = words[5][1:-2]
                                t = float_no_valueerror(words[0])
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
                                ) + float_no_valueerror(date[-3:])

                            if ": starting execution. mem usage" in line:
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
                            duration = float_no_valueerror(words[-2])
                            works_ended.append(
                                {"name": name, "key": key, "duration": duration}
                            )
                    print("\rdone" + 20 * " ")

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
                [times.min(), times.max()], [d, d], colors[i] + ":", linewidth=2
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
