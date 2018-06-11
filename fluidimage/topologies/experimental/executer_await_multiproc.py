"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""

from .executer_base import ExecuterBase
from .nb_workers import nb_max_workers


class ExecuterAwaitMultiprocs(ExecuterBase):
    def __init__(self, topology):
        super().__init__(topology)
