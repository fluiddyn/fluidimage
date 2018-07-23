import os
import json
import time
import functools
import scipy.io
import asyncio

from fluidimage import ParamContainer, SerieOfArraysFromFiles, SeriesOfArrays

from fluidimage.topologies.piv import TopologyPIV
from fluidimage.util.util import imread
from fluidimage.works.piv.multipass import WorkPIV
from fluidimage.data_objects.piv import ArrayCouple
from fluidimage.util.util import logger


class AsyncPIV:
    def __init__(self, params, work):

        self.params = params
        self.path_images = os.path.join(params.series.path)
        images_dir_name = self.params.series.path.split("/")[-1]
        self.saving_path = os.path.join(
            os.path.dirname(params.series.path),
            str(images_dir_name) + "." + params.saving.postfix,
        )
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        self.work = work
        self.loop = asyncio.get_event_loop()

        self._log_file = None
        self.img_tmp = None

    async def process(self, im1, im2, serie):
        """
        Call load_image, compute piv and save_piv with awaits
        :param name of im1
        :type str
        :param name of im2
        :type str
        :return: none
        """
        start = time.time()
        couple = await self.load_images(im1, im2, serie)
        result = await self.compute(couple)
        end = time.time()
        logger.info("Computed Image {}  : {}s".format(couple.name, end - start))
        await self.save_piv(result, im1, im2)
        end = time.time()
        logger.info(
            "finished Image {}  : {}s".format(im1 + " - " + im2, end - start)
        )
        return

    async def load_images(self, im1, im2, serie):
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
                None, functools.partial(imread, self.path_images + "/" + im1)
            )
        else:
            image1 = self.img_tmp

        image2 = await self.loop.run_in_executor(
            None, functools.partial(imread, self.path_images + "/" + im2)
        )
        params_mask = self.params.mask
        couple = ArrayCouple(
            names=(im1, im2),
            arrays=(image1, image2),
            params_mask=params_mask,
            serie=serie,
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
        return self.work.calcul(couple)

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
        result.save(
            path=self.saving_path + "/" + im1[:-4] + "_" + im2[:-4] + ".h5",
            kind="one",
        )

    def fill_queue(self, serie):
        """
        Define a concurenced work which is destined to be compute in a single process
        :param listdir: list of image names to compute
        :type list
        :return:
        """
        tasks = []
        serie_names = serie.get_name_all_arrays()
        print(serie_names)
        for i in range(len(serie_names) - 1):
            a_serie = serie.get_next_serie()
            tasks.append(
                asyncio.ensure_future(
                    self.process(serie_names[i], serie_names[i + 1], a_serie)
                )
            )
        self.loop.run_until_complete(asyncio.wait(tasks))
        self.loop.close()
