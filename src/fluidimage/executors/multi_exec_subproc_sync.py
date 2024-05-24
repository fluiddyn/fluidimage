"""Multi executor based on subprocesses using exec_seq_for_multi

.. autoclass:: MultiExecutorSubprocSync
   :members:
   :private-members:

"""

from fluidimage.executors.multi_exec_subproc import MultiExecutorSubproc


class MultiExecutorSubprocSync(MultiExecutorSubproc):
    executor_for_multi = "exec_seq_for_multi"


Executor = MultiExecutorSubprocSync
