import os
import gc
import multiprocessing as mp

import psutil

import numpy as np

process = psutil.Process(os.getpid())


def print_memory_usage():
    """Return the memory usage in Mo."""
    gc.collect()
    mem = process.memory_info()[0] / float(2 ** 20)
    print(f"Memory usage: {mem} Mo")
    return mem


def work(a):
    return 2 * a


def calcul_comm(work, obj_in, q):
    result = work(obj_in)
    q.put(result)


def main(nb_proc, size):
    processes = []
    queues = []
    for i in range(nb_proc):
        a = np.ones(size)
        q = mp.Queue()
        p = mp.Process(target=calcul_comm, args=(work, a, q))
        processes.append(p)
        queues.append(q)
        p.start()

    for q in queues:
        result = q.get()
        # del result

    for p in processes:
        p.join(1)

    print_memory_usage()


if __name__ == "__main__":

    for i in range(10):
        main(2, 100_000)
