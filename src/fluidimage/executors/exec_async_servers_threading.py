from .exec_async_servers import ExecutorAsyncServers


class ExecutorAsyncServersThreading(ExecutorAsyncServers):
    """Just used to get a better coverage"""

    _type_server = "threading"


Executor = ExecutorAsyncServersThreading
