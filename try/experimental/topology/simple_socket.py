# echo_server.py
import socket
import pickle
from fluidimage.works.piv import WorkPIV

host = "localhost"  # Symbolic name meaning all available interfaces
port = 8891  # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
while True:
    s.listen(1)
    conn, addr = s.accept()
    print("Connected by", addr)
    data = []
    while True:
        packet = conn.recv(4096)
        print("packet")
        if not packet:
            print("break")
            break
        data.append(packet)
    print("end receiving")
    data_arr = pickle.loads(b"".join(data))
    print(data_arr)
    data_ret = pickle.dumps(data_arr)
    print("about to send")
    s.sendall(data_ret)
    conn.close()
