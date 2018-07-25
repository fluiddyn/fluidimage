"""Topologies representing asynchronous computations
====================================================

A topology is responsible for the organization of the processing of a
succession of input "unit" objects. It contains the description as a "topology"
of the processing of one "unit" object (for example the production of 1 PIV
field from a couple of images). A topology is formed of a set of unit processes
linked by waiting queues.

The Topology object also organizes the "loop" over the input set. It organizes
the asynchronous (parallel) calls of the work units of the topology with the
correct transfer of data between them.

Users are particularly concerned with the PIV and preprocessing topologies:

.. autosummary::
   :toctree:

   piv
   bos
   preproc
   image2image
   surface_tracking

These others modules defined classes and functions useful for developers.

.. autosummary::
   :toctree:

   base
   waiting_queues
   log
   launcher

"""

import os

from fluiddyn.io.query import query


def prepare_path_dir_result(
    path_dir_input, path_saving, postfix_saving, how_saving
):
    """Makes new directory for results, if required, and returns its path."""

    if path_saving is not None:
        path_dir_result = path_saving
    else:
        path_dir_result = path_dir_input + "." + postfix_saving

    how = how_saving
    if os.path.exists(path_dir_result):
        if how == "ask":
            answer = query(
                "The directory {} ".format(path_dir_result)
                + "already exists. What do you want to do?\n"
                "New dir, Complete, Recompute or Stop?\n"
            )

            while answer.lower() not in ["n", "c", "r", "s"]:
                answer = query(
                    "The answer should be in ['n', 'c', 'r', 's']\n"
                    "Please type your answer again...\n"
                )

            if answer == "s":
                raise ValueError("Stopped by the user.")

            elif answer == "n":
                how = "new_dir"
            elif answer == "c":
                how = "complete"
            elif answer == "r":
                how = "recompute"

        if how == "new_dir":
            i = 0
            while os.path.exists(path_dir_result + str(i)):
                i += 1
            path_dir_result += str(i)

    if not os.path.exists(path_dir_result):
        os.mkdir(path_dir_result)

    return path_dir_result, how
