"""Executors of computational topologies
========================================

The executors are used to execute a topology.

From the user point of view, the executor is chosen from the method
:func:`fluidimage.topologies.base.TopologyBase.compute`. The default executor
is :class:`fluidimage.executors.multi_exec_async.MultiExecutorAsync`.

There are many executors with different computational strategies. Depending on
the computational topology and the hardware, it can be more efficient to chose
an executor compared to another.

.. autosummary::
   :toctree:

   base
   exec_sequential
   exec_async
   exec_async_sequential
   multi_exec_async
   multi_exec_subproc
   exec_async_seq_for_multi
   exec_async_multiproc
   exec_async_servers
   servers

.. autofunction:: get_entry_points

.. autofunction:: get_executor_names

.. autofunction:: import_executor_class

"""

import importlib
import os
import sys

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points

import trio


def afterfork():
    trio._core._thread_cache.THREAD_CACHE._idle_workers.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=afterfork)

from .base import ExecutorBase

# on Windows (and MacOS), one cannot use "multi_exec_async"
# because the OS does not support forks (or not fully) and
# multiprocessing works differently than on Linux
# see https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
supported_multi_executors = ["multi_exec_subproc"]
if sys.platform == "linux":
    supported_multi_executors.insert(0, "multi_exec_async")


_entry_points = None


def get_entry_points(reload=False):
    """Discover the executors installed"""
    global _entry_points
    if _entry_points is None or reload:
        _entry_points = entry_points(group="fluidimage.executors")

    if not _entry_points:
        raise RuntimeError(
            "No executor entry point were found, which indicates an installation issue."
        )

    return _entry_points


def get_executor_names():
    """Get available executor names"""
    return set(entry_point.name for entry_point in get_entry_points())


def _get_module_fullname_from_name(name):
    """Get the module name from an executor name

    Parameters
    ----------

    name : str
      Name of an executor.

    """
    entry_points = get_entry_points()
    selected_entry_points = entry_points.select(name=name)
    if len(selected_entry_points) == 0:
        raise ValueError(f"Cannot find an executor for {name = }. {entry_points}")
    elif len(selected_entry_points) > 1:
        logging.warning(
            f"{len(selected_entry_points)} plugins were found for {name = }"
        )

    return selected_entry_points[name].value


def import_executor_class(name):
    """Import an executor class.

    Parameters
    ----------

    name : str
      Executor name.

    Returns
    -------

    The corresponding executor class.

    """

    if isinstance(name, EntryPoint):
        module_fullname = name.value
        name = name.name
    else:
        module_fullname = _get_module_fullname_from_name(name)

    mod = importlib.import_module(module_fullname)
    return mod.Executor


__all__ = ["ExecutorBase", "get_executor_names", "import_executor_class"]
