"""

export OMP_NUM_THREADS=1

pythran cpubounded.py -o cpubounded_pythran.so

"""


from time import time

import trio

from cpubounded import long_func1
from cpubounded_pythran import long_func1

long_func = long_func1


n = 1_000_000_000
nb_proc = 4

t_start = time()
for iproc in range(nb_proc):
    print(f"start {iproc}: {time()-t_start:.4f} s")
    long_func(n)
    print(f"end {iproc}:   {time()-t_start:.4f} s")

duration_seq = time() - t_start
print(f"sequentially: {duration_seq:.4f} s")

t_start = time()

i_task_global = 0


async def calcul():
    global i_task_global
    i_task = i_task_global
    i_task_global += 1
    print(f"start {i_task}: {time()-t_start:.4f} s")
    await trio.run_sync_in_worker_thread(long_func, n)
    print(f"end {i_task}:   {time()-t_start:.4f} s")


async def main():
    async with trio.open_nursery() as nursery:
        for iproc in range(nb_proc):
            nursery.start_soon(calcul)


trio.run(main)

duration_par = time() - t_start
print(f"parallel: {duration_par:.4f} s")

print(f"speedup: {duration_seq/duration_par}")
