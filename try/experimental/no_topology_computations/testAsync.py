"""testPiv mode asynchrone
================
"""
import sys
import os
import time
import multiprocessing
import functools
import scipy.io
import asyncio

from fluidimage.topologies.piv import TopologyPIV
from fluidimage.util.util import imread
from fluidimage.works.piv.multipass import WorkPIV
from fluidimage.data_objects.piv import ArrayCouple

from fluidimage.util.util import logger
from fluidimage import config_logging
from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile


class AsyncPiv:
    def __init__(self, path_image, path_save):

        self.path_images = path_image
        self.path_save = path_save
        self.img_tmp = None
        self.params = TopologyPIV.create_default_params()
        self.workpiv = WorkPIV(self.params)
        self.params.series.path = self.path_images
        self.params.series.ind_start = 1

        self.params.piv0.shape_crop_im0 = 32
        self.params.multipass.number = 2
        self.params.multipass.use_tps = True

        self.params.multipass.use_tps = True

        # params.saving.how has to be equal to 'complete' for idempotent jobs
        self.params.saving.how = "complete"
        self.params.saving.postfix = "piv_complete_async"

        self.loop = asyncio.get_event_loop()

        self._log_file = None

    async def process(self, im1, im2):
        """
        Call load_image, compute piv and save_piv with awaits
        :param name of im1
        :type str
        :param name of im2
        :type str
        :return: none
        """
        start = time.time()
        couple = await self.load_images(im1, im2)
        result = await self.compute(couple)
        end = time.time()
        logger.info("Computed Image {}  : {}s".format(couple.name, end - start))
        await self.save_piv(result, im1, im2)
        end = time.time()
        logger.info(
            "finished Image {}  : {}s".format(im1 + " - " + im2, end - start)
        )
        return

    async def load_images(self, im1, im2):
        """
        load two images and make a couple
        :param name of im1
        :type str
        :param name of im2
        :type str
        :return: couple
        """
        start = time.time()
        if self.img_tmp == None:
            image1 = await self.loop.run_in_executor(
                None, functools.partial(imread, self.path_images + im1)
            )
        else:
            image1 = self.img_tmp

        image2 = await self.loop.run_in_executor(
            None, functools.partial(imread, self.path_images + im2)
        )
        params_mask = self.params.mask
        couple = ArrayCouple(
            names=(im1, im2), arrays=(image1, image2), params_mask=params_mask
        )
        self.img_tmp = image2
        end = time.time()
        logger.info(
            "Loaded Image {}  : {}s".format(im1 + " - " + im2, end - start)
        )
        return couple

    async def compute(self, couple):
        """
        Create a pivwork and compute a couple
        :param couple: a couple from arrayCouple
        :type ArrayCouple
        :return: a piv object
        """
        start = time.time()
        workpiv = WorkPIV(self.params)
        end = time.time()
        return workpiv.calcul(couple)

    async def save_piv(self, result, im1, im2):
        """
        Save the light result at path_save
        :param result of the computing
        :param name of im1
        :type str
        :param name of im2
        :type str
        :return:
        """
        light_result = result.make_light_result()
        im1 = im1[:-4]
        im2 = im2[:-4]
        scipy.io.savemat(
            self.path_save + "piv_" + im1 + "_" + im2,
            mdict={
                "deltaxs": light_result.deltaxs,
                "deltays": light_result.deltays,
                "xs": light_result.xs,
                "ys": light_result.ys,
            },
        )

    def a_process(self, listdir):
        """
        Define a concurenced work which is destined to be compute in a single process
        :param listdir: list of image names to compute
        :type list
        :return:
        """
        tasks = []
        for i in range(len(listdir) - 1):
            tasks.append(
                asyncio.ensure_future(self.process(listdir[i], listdir[i + 1]))
            )
        self.loop.run_until_complete(asyncio.wait(tasks))
        self.loop.close()


def main():
    # Define path
    sub_path_image = "Images2"
    path_save = "../../image_samples/Karman/{}.results.async/".format(
        sub_path_image
    )
    # Logger
    log = os.path.join(
        path_save, "log_" + time_as_str() + "_" + str(os.getpid()) + ".txt"
    )
    log_file = open(log, "w")
    sys.stdout = MultiFile([sys.stdout, log_file])
    config_logging("info", file=sys.stdout)
    # Managing dir paths
    path = f"../../image_samples/Karman/{sub_path_image}/"
    assert os.listdir(path)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    def partition(lst, n):
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

    nb_process = multiprocessing.cpu_count()
    # spliting images list
    listdir = os.listdir(path)

    if len(listdir) <= nb_process:  # if there is less piv to compute than cpu
        nb_process = len(listdir) - 1  # adapt process number
    print(f"nb process :{nb_process}")
    listdir.sort()
    listdir = partition(listdir, nb_process)
    # making and starting processes
    processes = []
    for i in range(nb_process):
        async_piv = AsyncPiv(path, path_save)
        p = multiprocessing.Process(
            target=async_piv.a_process, args=(listdir[i],)
        )
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
