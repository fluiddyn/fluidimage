"""

- IO tasks are handled with an asyncio event loop.

- CPU/GPU bounded tasks are handles with microservices (r2py, mpi4py, ?)

"""

from .executer_base import ExecuterBase
from .nb_workers import nb_max_workers


class ExecuterAwaitMicroservices(ExecuterBase):
    def __init__(self, topology):
        super().__init__(topology)
