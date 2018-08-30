import unittest
from shutil import rmtree
from pathlib import Path

from fluiddyn.io import stdout_redirected

from fluidimage.experimental.topologies.example import TopologyExample
from fluidimage.experimental.executors.executor_await import (
    ExecutorAwaitMultiprocs
)

from fluidimage import path_image_samples

path_input = path_image_samples / "Karman/Images"


class TestPivNew(unittest.TestCase):

    def tearDown(self):
        path_out = self.topology.path_dir_result
        if path_out.exists():
            rmtree(path_out)

    def test_piv_new(self):

        with stdout_redirected():
            self.topology = TopologyExample(
                path_input,
                logging_level="info"
            )

            executer = ExecutorAwaitMultiprocs(
                self.topology,
                multi_executor=False,
                sleep_time=0.1,
                worker_limit=4,
                queues_limit=5,
            )

            self.topology.compute(executer)
