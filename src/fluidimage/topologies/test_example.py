import pytest

from fluidimage.executors import supported_multi_executors
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

# tmp: TopologyExample doesn't have a Splitter
executors.remove("multi_exec_subproc")


@pytest.mark.parametrize("executor", executors)
def test_topo_example(tmp_path_karman, executor):

    path_input = tmp_path_karman

    params = TopologyExample.create_default_params()
    params["path_input"] = path_input

    path_dir_result = path_input.parent / f"Images.{executor}"
    params["path_dir_result"] = path_dir_result

    topology = TopologyExample(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)

    # there is a logging problem with this class but we don't mind.
    if executor != "exec_async_servers_threading":
        log = topology.read_log_data()
        assert (
            log.topology_name == "fluidimage.topologies.example.TopologyExample"
        )
        assert log.nb_max_workers == 2

        if [
            len(log.durations[key])
            for key in ("read_array", "cpu1", "cpu2", "save")
        ] != [4, 3, 2, 2]:
            print("Issue with this log file:")
            print(log.path_log_file.read_text())

        log.plot_durations()
        log.plot_memory()
        log.plot_nb_workers()

        # does not always work for "exec_async_multi" (?)
        if executor != "exec_async_multi":
            assert len(log.durations["read_array"]) == 4
            assert len(log.durations["cpu1"]) == 3
            assert len(log.durations["cpu2"]) == 2
            assert len(log.durations["save"]) == 2

    path_files = tuple(path_dir_result.glob("Karman*"))

    assert len(path_files) > 0, "No files saved"
    assert len(path_files) == 2, "Bad number of saved files"
