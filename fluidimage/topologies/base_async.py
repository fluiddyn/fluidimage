import math
import sys
import os
import time
import multiprocessing


from fluidimage import config_logging
from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile
from .. import SerieOfArraysFromFiles, SeriesOfArrays

class Base_async:

    def __init__(self, params, work, async_proc_class,logging_level):

        self.params = params
        self.async_process_class = async_proc_class
        self.images_path = os.path.join(params.series.path, params.path.sub_images_path)
        self.saving_path = os.path.join(params.series.path,params.saving.postfix)
        self.series = []
        self. processes = []
        self.async_process = []
        self.work = work

        #Logger
        log = os.path.join(
            self.saving_path,
            "log_" + time_as_str() + "_" + str(os.getpid()) + ".txt",
        )
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        log_file = open(log, "w")
        sys.stdout = MultiFile([sys.stdout, log_file])
        config_logging("info", file=sys.stdout)
        # Managing dir paths
        assert os.listdir(self.images_path)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

    def compute(self):
        self._make_processes()
        self._start_processes()

    def _make_processes(self):
        nb_process = multiprocessing.cpu_count()
        # splitting series
        serie_arrays = SerieOfArraysFromFiles(path=self.images_path)
        ind_start = self.params.series.ind_start
        ind_stop = self.params.series.ind_stop
        ind_step = self.params.series.ind_step

        print("Nb process : {}".format(nb_process))
        nb_image = ind_stop - ind_start +1
        cut = int(nb_image/nb_process)
        rest = nb_image % nb_process
        print("nb images : "+str(nb_image))
        print("cut : "+str(cut))
        print("reste : "+str(rest))
        print(nb_process * cut + rest)

        for i in range(nb_process):

            if rest > 0:
                plus = 1
            else :
                plus = 0
            self.series.append(SeriesOfArrays(
                serie_arrays,
                self.params.series.strcouple,
                ind_start=ind_start,
                ind_stop= ind_start + cut + plus,
                ind_step=ind_step,
            ))
            ind_start = ind_start + cut + plus
            rest -= 1

            print(self.series[i].get_name_all_arrays())
        # making and starting processes
        for i in range(nb_process):
            async_piv = self.async_process_class(self.params, self.work)
            self.processes.append(multiprocessing.Process(target=async_piv.a_process, args=(self.series[i],)))


    def _start_processes(self):
        for p in self.processes:
            p.start()


    def _partition(self, lst, n):
        """
        Partition evently lst into n sublists and
        add the last images of each sublist to the head
        of the next sublist ( in order to compute all piv )
        :param lst: a list
        :param n: number of sublist wanted
        :return: A sliced list
        """
        L = len(lst)
        assert 0 < n <= L
        s, r = divmod(L, n)
        t = s + 1
        lst = [lst[p : p + t] for p in range(0, r * t, t)] + [
            lst[p : p + s] for p in range(r * t, L, s)
        ]
        #  in order to compute all piv
        #  add the last images of each sublist to the head of the next sublist
        for i in range(1, n):
            lst[i].insert(0, lst[i - 1][-1])
        return lst