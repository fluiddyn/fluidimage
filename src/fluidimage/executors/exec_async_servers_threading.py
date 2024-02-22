"""Executor async/await using servers
=====================================

.. autoclass:: ExecutorAsyncServersThreading
   :members:
   :private-members:

"""

from .exec_async_servers import ExecutorAsyncServers


class ExecutorAsyncServersThreading(ExecutorAsyncServers):
    """Just used to get a better coverage"""

    _type_server = "threading"


Executor = ExecutorAsyncServersThreading
