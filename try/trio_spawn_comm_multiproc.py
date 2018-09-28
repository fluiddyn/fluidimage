from multiprocessing import Process, Queue

import trio

import numpy as np


async def sleep():
    print("enter sleep1s")
    await trio.sleep(0.2)
    print("end sleep1s")


def cpu_bounded_task(input_data):
    result = input_data.copy()
    for i in range(1000000 - 1):
        result += input_data
    return result


def server(q_c2s, q_s2c):
    async def main_server():
        # get the data to be processed
        input_data = await trio.run_sync_in_worker_thread(q_c2s.get)
        print("in server: input_data received", input_data)
        # a CPU-bounded task
        result = cpu_bounded_task(input_data)
        print("in server: sending back the answer", result)
        await trio.run_sync_in_worker_thread(q_s2c.put, result)

    trio.run(main_server)


async def client(q_c2s, q_s2c):
    input_data = np.arange(10)
    print("in client: sending the input_data", input_data)
    await trio.run_sync_in_worker_thread(q_c2s.put, input_data)
    result = await trio.run_sync_in_worker_thread(q_s2c.get)
    print("in client: result received", result)


async def parent(q_c2s, q_s2c):

    async with trio.open_nursery() as nursery:
        nursery.start_soon(sleep)
        nursery.start_soon(client, q_c2s, q_s2c)
        nursery.start_soon(sleep)


def main():
    q_c2s = Queue()
    q_s2c = Queue()
    p = Process(target=server, args=(q_c2s, q_s2c))
    p.start()
    trio.run(parent, q_c2s, q_s2c)
    p.join()


if __name__ == "__main__":
    main()
