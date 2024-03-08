import pytest

from fluidimage.executors import supported_multi_executors
from fluidimage.topologies import LogTopology
from fluidimage.topologies.example import TopologyExample

executors = [
    "exec_sequential",
    "exec_async_sequential",
    "exec_async",
    "exec_async_multi",
    "exec_async_servers",
    "exec_async_servers_threading",
]

executors.extend(supported_multi_executors)


@pytest.mark.parametrize("executor", supported_multi_executors)
def test_topo_example(tmp_path_karman, executor):

    path_input = tmp_path_karman

    params = TopologyExample.create_default_params()
    params["path_input"] = path_input

    path_dir_result = path_input.parent / f"Images.{executor}"
    params["path_dir_result"] = path_dir_result

    topology = TopologyExample(params, logging_level="debug")
    topology.compute(executor, nb_max_workers=2)

    if executor != "exec_async_servers_threading":
        # there is a logging problem with this class but we don't mind.
        log = LogTopology(path_dir_result)
        assert log.topology_name is not None

    path_files = tuple(path_dir_result.glob("Karman*"))

    assert len(path_files) > 0, "No files saved"
    assert len(path_files) == 2, "Bad number of saved files"
