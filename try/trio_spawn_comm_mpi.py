import sys
from functools import partial

import trio

import numpy as np
from mpi4py import MPI


async def sleep():
    print("enter sleep")
    await trio.sleep(0.2)
    print("end sleep")


def cpu_bounded_task(input_data):
    print("cpu_bounded_task starting")
    result = input_data.copy()
    for i in range(1000000 - 1):
        result += input_data
    print("cpu_bounded_task finished ")
    return result


if "server" not in sys.argv:
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable, args=["trio_spawn_comm_mpi.py", "server"]
    )

    async def client():
        input_data = np.arange(4)
        print("in client: sending the input_data", input_data)
        send = partial(comm.send, dest=0, tag=0)
        await trio.run_sync_in_worker_thread(send, input_data)

        print("in client: recv")
        recv = partial(comm.recv, tag=1)
        result = await trio.run_sync_in_worker_thread(recv)
        print("in client: result received", result)

    async def parent():
        async with trio.open_nursery() as nursery:
            nursery.start_soon(sleep)
            nursery.start_soon(client)
            nursery.start_soon(sleep)

    trio.run(parent)

    print("in client, end")
    comm.barrier()

else:
    comm = MPI.Comm.Get_parent()

    async def main_server():
        # get the data to be processed
        recv = partial(comm.recv, tag=0)
        input_data = await trio.run_sync_in_worker_thread(recv)
        print("in server: input_data received", input_data)
        # a CPU-bounded task
        result = cpu_bounded_task(input_data)
        print("in server: sending back the answer", result)
        send = partial(comm.send, dest=0, tag=1)
        await trio.run_sync_in_worker_thread(send, result)

    trio.run(main_server)
    comm.barrier()
