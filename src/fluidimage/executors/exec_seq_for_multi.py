from fluidimage.executors.exec_async_seq_for_multi import ExecutorAsyncSeqForMulti
from fluidimage.executors.exec_sequential import ExecutorSequential


class ExecutorSeqForMulti(ExecutorSequential, ExecutorAsyncSeqForMulti):
    """Sequential executor modified for multi executors"""
