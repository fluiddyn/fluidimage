"""Parse and analyze logging files.
===================================


"""
from __future__ import print_function

from glob import glob
import time
import os

import numpy as np
import matplotlib.pyplot as plt
plt.ion()


class LogTopology(object):
    """Parse and analyze logging files.

    """
    def __init__(self, path=None, path_dir=''):

        if path is None:
            pattern = os.path.join(path_dir, 'log_*.txt')
            paths = glob(pattern)
            if len(paths) == 0:
                raise ValueError(
                    'No log files found in the current directory.')
            path = paths[-1]

        self._parse_log(path)

    def _parse_log(self, path):
        self.works = works = []
        self.works_ended = works_ended = []
        with open(path, 'r') as f:
            print('Parsing log file: ', path)
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
                        work = words[4]
                        key = words[5][1:-1]
                        works.append({
                            'work': work, 'key': key, 'date': date,
                            'mem_start': mem, 'time': t - time_start})

                if line.startswith('INFO: work '):
                    words = line.split()
                    work = words[2]
                    key = words[3][1:-1]
                    duration = float(words[-2])
                    works_ended.append({
                        'work': work, 'key': key, 'duration': duration})

    def plot_memory(self):

        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('time (s)')
        ax.set_ylabel('memory (Mo)')

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
