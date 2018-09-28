import sys

import numpy as np
from mpi4py import MPI

if "server" not in sys.argv:
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable, args=["spawn_comm_mpi.py", "server"]
    )
    comm.send(np.arange(4), dest=0, tag=0)
    print("in client, end")
    comm.barrier()

else:
    comm = MPI.Comm.Get_parent()
    input_data = comm.recv(tag=0)
    print("in server, input_data =", input_data)
    comm.barrier()
