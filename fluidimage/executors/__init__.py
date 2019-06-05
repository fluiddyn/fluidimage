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
   exec_async_multiproc
   exec_async_servers
   servers

"""

from .base import ExecutorBase
from .exec_async import ExecutorAsync
from .exec_async_multiproc import ExecutorAsyncMultiproc
from .exec_async_sequential import ExecutorAsyncSequential
from .exec_async_servers import (
    ExecutorAsyncServers,
    ExecutorAsyncServersThreading,
)
from .exec_sequential import ExecutorSequential
from .multi_exec_async import MultiExecutorAsync

executors = {
    "exec_sequential": ExecutorSequential,
    "exec_async": ExecutorAsync,
    "exec_async_sequential": ExecutorAsyncSequential,
    "multi_exec_async": MultiExecutorAsync,
    "exec_async_multi": ExecutorAsyncMultiproc,
    "exec_async_servers": ExecutorAsyncServers,
    "exec_async_servers_threading": ExecutorAsyncServersThreading,
}

__all__ = ["ExecutorBase", "executors"]
