"""Topologies representing asynchronous computations
====================================================

A topology represents an asynchronous computation as a graph of **waiting
queues** and **works**. Works take (an) item(s) in (an) input queue(s), process
it/them and put the result in (an) output queue(s).

All topologies inherit from a base class
:class:`fluidimage.topologies.base.TopologyBase`, which has methods to define
the topology (``add_queue`` and ``add_work``), to represent the computational
graph (``make_code_graphviz``) and finally to execute it (``compute``).

Users are particularly concerned with the following already defined topologies:

.. autosummary::
   :toctree:

   piv
   bos
   preproc
   image2image
   surface_tracking
   optical_flow

These others modules defined classes and functions useful for developers.

.. autosummary::
   :toctree:

   base
   log
   launcher

"""

import os
import sys
from pathlib import Path

from fluiddyn.io.query import query

from .base import TopologyBase
from .log import LogTopology

__all__ = ["LogTopology", "TopologyBase"]

how_values = ("ask", "new_dir", "complete", "recompute")


def prepare_path_dir_result(
    path_dir_input, path_saving, postfix_saving, how_saving
):
    """Makes new directory for results, if required, and returns its path."""

    if how_saving not in how_values:
        raise ValueError(
            f"how_saving (here equal to '{how_saving}') "
            f"should be in {how_values}"
        )

    path_dir_input = str(path_dir_input)

    if path_saving is not None:
        path_dir_result = path_saving
    else:
        path_dir_result = path_dir_input + "." + postfix_saving

    how = how_saving
    if os.path.exists(path_dir_result):
        if how == "ask":
            answer = query(
                f"The directory {path_dir_result} "
                + "already exists. What do you want to do?\n"
                "New dir, Complete, Recompute or Stop?\n"
            )

            while answer.lower() not in ["n", "c", "r", "s"]:
                answer = query(
                    "The answer should be in ['n', 'c', 'r', 's']\n"
                    "Please type your answer again...\n"
                )

            if answer == "s":
                print("Stopped by the user.")
                sys.exit()

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

    path_dir_result = Path(path_dir_result)
    path_dir_result.mkdir(exist_ok=True)
    return path_dir_result, how
