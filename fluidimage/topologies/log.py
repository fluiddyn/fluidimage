"""Parse and analyze logging files.
===================================


"""
from __future__ import print_function

from glob import glob
import os
import time

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

colors = ['r', 'b', 'y', 'g']


class LogTopology(object):
    """Parse and analyze logging files.

    """
    def __init__(self, path=None):

        if path is None:
            paths = glob('log_*.txt')
            paths.sort()
            if len(paths) == 0:
                raise ValueError(
                    'No log files found in the current directory.')
            path = paths[-1]

        self.log_file = os.path.split(path)[-1]

        self._parse_log(path)

    def _parse_log(self, path):
        self.works = works = []
        self.works_ended = works_ended = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('INFO: ') and '. mem usage: ' in line:
                    words = line.split()
                    date = words[1][5:-1]
                    t = time.mktime(
                        time.strptime(date[:-3], '%Y-%m-%d_%H-%M-%S')) + \
                        float(date[-3:])
                    try:
                        mem = float(words[-2])
                    except ValueError:
                        pass
                    if 'start compute. mem usage:' in line:
                        self.date_start = date
                        self.mem_start = mem
                        time_start = t
                    elif ': end of `compute`. mem usage' in line:
                        self.date_end = date
                        self.duration = t - time_start
                        self.mem_end = mem
                    elif ': launch work ' in line:
                        name = words[4]
                        key = words[5][1:-2]
                        works.append({
                            'name': name, 'key': key, 'date': date,
                            'mem_start': mem, 'time': t - time_start})

                if line.startswith('INFO: work '):
                    words = line.split()
                    name = words[2]
                    key = words[3][1:-1]
                    duration = float(words[-2])
                    works_ended.append({
                        'name': name, 'key': key, 'duration': duration})

                self.names_works = names_works = []
                for work in works:
                    if work['name'] not in names_works:
                        names_works.append(work['name'])

        self.durations = durations = {}
        self.times = times = {}
        self.keys = keys = {}
        for name in self.names_works:
            times[name] = []
            keys[name] = []
            for work in self.works:
                if work['name'] == name:
                    times[name].append(work['time'])
                    keys[name].append(work['key'])

            durations[name] = []
            for key in keys[name]:
                founded = False
                for work in self.works_ended:
                    if work['name'] == name and work['key'] == key:
                        durations[name].append(work['duration'])
                        founded = True
                        break
                if not founded:
                    durations[name].append(np.nan)

    def plot_memory(self):

        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('time (s)')
        ax.set_ylabel('memory (Mo)')
        ax.set_title(self.log_file)

        memories = np.empty(len(self.works))
        times = np.empty(len(self.works))
        for i, work in enumerate(self.works):
            memories[i] = work['mem_start']
            times[i] = work['time']

        ax.plot(times, memories, 'o-')
        ax.plot(0, self.mem_start, 'x')
        if hasattr(self, 'duration'):
            ax.plot(self.duration, self.mem_end, 'x')
        plt.show()

    def plot_durations(self):

        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('time (s)')
        ax.set_ylabel('duration (s)')
        ax.set_title(self.log_file)

        lines = []

        for i, name in enumerate(self.names_works):
            times = np.array(self.times[name])
            durations = self.durations[name]
            l, = ax.plot(times, durations, colors[i] + 'o')
            lines.append(l)

            for it, t in enumerate(times):
                d = durations[it]
                ax.plot([t, t+d], [d, d], colors[i])

            d = np.nanmean(durations)
            ax.plot([times.min(), times.max()], [d, d], colors[i] + '-',
                    linewidth=2)

        ax.legend(lines, self.names_works, loc='center left',
                  fontsize='x-small')

        plt.show()

    def plot_nb_workers(self, str_names=None):
        if str_names is not None:
            names = [name for name in self.names_works if str_names in name]
        else:
            names = self.names_works

        nb_workers = {}
        times = {}
        for name in names:
            times_start = list(self.times[name])
            times_stop = list(np.array(times_start) +
                              np.array(self.durations[name]))

            deltas = np.array([1 for t in times_start] +
                              [-1 for t in times_stop])
            times_unsorted = np.array(times_start + times_stop)

            argsort = np.argsort(times_unsorted)

            ts = times_unsorted[argsort]
            nbws = np.cumsum(deltas[argsort])

            ts2 = []
            nbws2 = []
            for i, t in enumerate(ts[:-1]):
                nbw = nbws[i]
                ts2.append(t)
                ts2.append(ts[i+1])
                nbws2.append(nbw)
                nbws2.append(nbw)

            times[name] = ts2
            nb_workers[name] = nbws2

        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('time (s)')
        ax.set_ylabel('number of workers')
        ax.set_title(self.log_file)

        lines = []

        for i, name in enumerate(self.names_works):
            l, = ax.plot(times[name], nb_workers[name], colors[i] + '-')
            lines.append(l)

        ax.legend(lines, names, loc='center left', fontsize='x-small')

        plt.show()
