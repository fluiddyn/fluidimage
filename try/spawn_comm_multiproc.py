from multiprocessing import Process, Queue

import numpy as np


def f(q):
    q.put(np.arange(4))


if __name__ == "__main__":
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())  # prints "[42, None, 'hello']"
    p.join()
