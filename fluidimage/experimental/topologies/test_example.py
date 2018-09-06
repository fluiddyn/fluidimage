import unittest
from shutil import rmtree

from fluidimage.experimental.topologies.example import TopologyExample

from fluidimage import path_image_samples

path_input = path_image_samples / "Karman/Images"


class TestTopoExample(unittest.TestCase):
    def tearDown(self):
        for topology in self.topologies:
            path_out = topology.path_dir_result
            if path_out.exists():
                rmtree(path_out)

    def test_example(self):

        self.topologies = []
        executors = [
            None,
            "exec_async_sequential",
            "multi_exec_async",
            "exec_async_multi",
            "exec_async_servers",
            "exec_async_servers_threading",
        ]
        params = TopologyExample.create_default_params()
        params["path_input"] = path_input

        for executor in executors:
            path_dir_result = path_input.parent / f"Images.{executor}"
            params["path_dir_result"] = path_dir_result

            topology = TopologyExample(params, logging_level="debug")
            self.topologies.append(topology)
            topology.compute(executor, nb_max_workers=2)


if __name__ == "__main__":
    unittest.main()
