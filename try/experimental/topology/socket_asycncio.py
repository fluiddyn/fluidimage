import trio, pickle
from fluidimage.works.piv.multipass import WorkPIV
import json
import sys
import random
from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage.topologies import image2image

# async def handle_client(client):
#     packet = None
#     data = []
#     while True:
#         print("packet")
#         packet = (await loop.sock_recv(client, 4096))
#         if not packet:
#             break
#         data.append(packet)
#     data = pickle.loads(b"".join(data))
#     print(data)
#
#     # await loop.sock_sendall(client, response.encode('utf8'))
#     client.close()
def create_default_params():
    """Class method returning the default parameters.

    For developers: cf. fluidsim.base.params

    """
    params = ParamContainer(tag="params")

    params._set_child(
        "series",
        attribs={
            "path": "",
            "strcouple": "i:i+2",
            "ind_start": 0,
            "ind_stop": None,
            "ind_step": 1,
        },
    )

    params.series._set_doc(
        """
Parameters indicating the input series of images.

path : str, {''}

String indicating the input images (can be a full path towards an image
file or a string given to `glob`).

strcouple : 'i:i+2'

String indicating as a Python slicing how couples of images are formed.
There is one couple per value of `i`. The values of `i` are set with the
other parameters `ind_start`, `ind_step` and `ind_stop` approximately with
the function range (`range(ind_start, ind_stop, ind_step)`).

Python slicing is a very powerful notation to define subset from a
(possibly multidimensional) set of images. For a user, an alternative is to
understand how Python slicing works. See for example this page:
http://stackoverflow.com/questions/509211/explain-pythons-slice-notation.

Another possibility is to follow simple examples:

For single-frame images (im0, im1, im2, im3, ...), we keep the default
value 'i:i+2' to form the couples (im0, im1), (im1, im2), ...

To see what it gives, one can use ipython and range:

>>> i = 0
>>> list(range(10))[i:i+2]
[0, 1]

>>> list(range(10))[i:i+4:2]
[0, 2]

We see that we can also use the value 'i:i+4:2' to form the couples (im0,
im2), (im1, im3), ...

For double-frame images (im1a, im1b, im2a, im2b, ...) you can write

>>> params.series.strcouple = 'i, 0:2'

In this case, the first couple will be (im1a, im1b).

To get the first couple (im1a, im1a), we would have to write

>>> params.series.strcouple = 'i:i+2, 0'

ind_start : int, {0}

ind_step : int, {1}

int_stop : None

"""
    )

    params._set_child(
        "saving", attribs={"path": None, "how": "ask", "postfix": "piv"}
    )

    params.saving._set_doc(
        """Saving of the results.

path : None or str

Path of the directory where the data will be saved. If None, the path is
obtained from the input path and the parameter `postfix`.

how : str {'ask'}

'ask', 'new_dir', 'complete' or 'recompute'.

postfix : str

Postfix from which the output file is computed.
"""
    )

    WorkPIV._complete_params_with_default(params)

    params._set_internal_attr(
        "_value_text",
        json.dumps(
            {
                "program": "fluidimage",
                "module": "fluidimage.topologies.piv",
                "class": "TopologyPIV",
            }
        ),
    )

    params._set_child("preproc")
    image2image.complete_im2im_params_with_default(params.preproc)

    return params


async def run_server():
    params = create_default_params()
    workpiv = WorkPIV(params)

    server = trio.socket.socket()
    await server.bind(("localhost", 8888))
    server.listen(80)
    while True:
        conn, adr = await server.accept()
        print(f"connection etablished from {adr}")
        data = []
        while True:
            packet = await conn.recv(4096)
            print("packet")
            if not packet:
                break
            data.append(packet)
        print("end receiving")
        data = pickle.loads(b"".join(data))
        try:
            data = workpiv.calcul(data)
            data = pickle.dumps(data)
            await conn.send(data)
        except:
            pass
        conn.shutdown(1)


async def start_server():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(run_server)


trio.run(start_server)
