"""Executors
============

The executors are used to execute a topology computation.

Each executor has a different way to compute it. Depending on the topology and the
hardware, it can be more efficient to chose an executor compared to another.

.. autosummary::
   :toctree:

   base
   exec_async
   multiexec_async
   exec_async_multiproc
   exec_async_servers

"""

from .base import ExecutorBase

from .exec_async import ExecutorAsync
from .multiexec_async import MultiExecutorAsync
from .exec_async_multiproc import ExecutorAsyncMultiproc


executors = {
    "exec_async": ExecutorAsync,
    "multi_exec_async": MultiExecutorAsync,
    "exec_async_multi": ExecutorAsyncMultiproc,
}

__all__ = ["ExecutorBase", "executors"]
