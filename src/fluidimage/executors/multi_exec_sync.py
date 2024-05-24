"""
Multi executors sync
=====================

.. autoclass:: MultiExecutorSync
   :members:
   :private-members:

"""

from fluidimage.executors.exec_seq_for_multi import ExecutorSeqForMulti
from fluidimage.executors.multi_exec_async import MultiExecutorAsync


class MultiExecutorSync(MultiExecutorAsync):
    ExecutorForMulti = ExecutorSeqForMulti


Executor = MultiExecutorSync
