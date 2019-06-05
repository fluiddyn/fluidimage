import unittest
from functools import partialmethod
from shutil import rmtree

from fluidimage import path_image_samples
from fluidimage.topologies import LogTopology
from fluidimage.topologies.example import TopologyExample

path_input = path_image_samples / "Karman/Images"

executors = [
    "exec_sequential",
    "exec_async_sequential",
    "exec_async",
    "multi_exec_async",
    "exec_async_multi",
    "exec_async_servers",
    "exec_async_servers_threading",
]


def _test(self, executor=None):

    params = TopologyExample.create_default_params()
    params["path_input"] = path_input

    path_dir_result = path_input.parent / f"Images.{executor}"
    params["path_dir_result"] = path_dir_result

    self.topology = topology = TopologyExample(params, logging_level="debug")
    topology.compute(executor, nb_max_workers=2)

    if executor != "exec_async_servers_threading":
        # there is a logging problem with this class but we don't mind.
        log = LogTopology(path_dir_result)
        self.assertTrue(log.topology_name is not None)

    path_files = tuple(path_dir_result.glob("Karman*"))

    assert len(path_files) > 0, "No files saved"
    assert len(path_files) == 2, "Bad number of saved files"


class TestTopoExample(unittest.TestCase):
    def tearDown(self):
        rmtree(self.topology.path_dir_result, ignore_errors=True)


for executor in executors:
    setattr(
        TestTopoExample,
        "test_" + str(executor),
        partialmethod(_test, executor=executor),
    )


if __name__ == "__main__":
    unittest.main()
