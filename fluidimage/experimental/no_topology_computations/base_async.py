import math
import sys
import os
import time
import multiprocessing


from fluidimage import config_logging
from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile
from fluidimage import SerieOfArraysFromFiles, SeriesOfArrays


class BaseAsync:
    def __init__(self, params, work, async_proc_class, logging_level="info"):

        self.params = params
        self.async_process_class = async_proc_class
        self.images_path = os.path.join(params.series.path)
        images_dir_name = self.params.series.path.split("/")[-1]
        self.saving_path = os.path.join(
            os.path.dirname(params.series.path),
            str(images_dir_name) + "." + params.saving.postfix,
        )
        self.series = []
        self.processes = []
        self.async_process = []
        self.work = work

        # Logger
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
        if (
            self.params.series.ind_stop - self.params.series.ind_start
        ) / self.params.series.ind_step < nb_process:
            nb_process = (
                self.params.series.ind_stop - self.params.series.ind_start
            ) + 1
        self._make_partition(serie_arrays, nb_process)
        # making and starting processes
        for i in range(nb_process):
            async_piv = self.async_process_class(self.params, self.work)
            self.processes.append(
                multiprocessing.Process(
                    target=async_piv.fill_queue, args=(self.series[i],)
                )
            )

    def _start_processes(self):
        for p in self.processes:
            p.start()
        for p in self.processes:
            p.join()

    def _make_partition(self, serie_arrays, n):
        """
        Partition a SerieOfArrayFromFile into n SeriesOfArray
        :param serie_arrays: A SerieOfArrayFromFile
        :type SerieOfArrayFromFile
        :param n: The number of slices
        :type int
        :return:
        """
        print("nb process = " + str(n))
        ind_start = self.params.series.ind_start
        ind_stop = self.params.series.ind_stop
        ind_step = self.params.series.ind_step

        nb_image = ind_stop - ind_start + 1
        cut = int(nb_image / n)
        rest = nb_image % n
        for i in range(n):
            if rest > 0:
                plus = 1
            else:
                plus = 0
            self.series.append(
                SeriesOfArrays(
                    serie_arrays,
                    self.params.series.strcouple,
                    ind_start=ind_start,
                    ind_stop=ind_start + cut + plus,
                    ind_step=ind_step,
                )
            )
            ind_start = ind_start + cut + plus
            rest -= 1
