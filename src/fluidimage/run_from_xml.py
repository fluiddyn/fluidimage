"""Uvmat interface
==================

From Matlab, something like::

  system(['python -m fluidimage.run_from_xml ' path_to_instructions_xml])


.. todo::

   UVmat civserie: input and output.

"""

import argparse
import os
from abc import ABC, abstractmethod
from time import time

from fluiddyn import time_as_str
from fluiddyn.util import import_class
from fluiddyn.util.paramcontainer import tidy_container
from fluidimage.topologies.piv import TopologyPIV

from . import (  # SerieOfArraysFromFiles,
    ParamContainer,
    config_logging,
    logger,
    reset_logger,
)

# import numpy as np
# import scipy


def tidy_uvmat_instructions(params):
    params._set_internal_attr("_tag", "instructions_uvmat")

    # get nicer names and a simpler organization...
    tidy_container(params)

    params.input_table = input_table = params.input_table.split(" & ")
    for i in range(1, len(input_table)):
        if input_table[i].startswith("/"):
            input_table[i] = input_table[i][1:]

    filename = "".join(input_table[2:])
    path_dir = os.path.join(*input_table[:2])
    path_file_input = os.path.abspath(os.path.join(path_dir, filename))
    path_dir_input = path_dir
    params._set_attrib("path_dir_input", path_dir_input)
    params._set_attrib("path_file_input", path_file_input)

    params._set_attrib("path_dir_output", path_dir_input + params.output_dir_ext)

    ir = params.index_range
    if not hasattr(ir, "incr_i"):
        ir._set_attrib("incr_i", 1)

    if not hasattr(ir, "first_i"):
        print("Warning: no attribute first_i in xml UVmat file.")
        ir._set_attrib("first_i", 1)

    if not hasattr(ir, "last_i"):
        print("Warning: no attribute last_i in xml UVmat file.")
        ir._set_attrib("last_i", None)

    if not hasattr(ir, "first_j"):
        str_subset = f"i:i+{ir.incr_i + 1}:{ir.incr_i}"
        ind_start = ir.first_i

        if ir.last_i is None:
            ind_stop = None
        else:
            ind_stop = ir.last_i - ir.incr_i + 1
    else:
        raise NotImplementedError

    params._set_attrib("str_subset", str_subset)
    params._set_attrib("ind_stop", ind_stop)
    params._set_attrib("ind_start", ind_start)


class ActionBase(ABC):

    @abstractmethod
    def __init__(self, instructions, args):
        """Initialize the action class"""

    @abstractmethod
    def compute(self):
        """Perform the computation"""


# class ActionAverage(ActionBase):
#     """Compute the average and save as a png file."""

#     def __init__(self, instructions, args):
#         self.instructions = instructions

#         # create the serie of arrays
#         logger.info("Create the serie of arrays")
#         self.serie_arrays = SerieOfArraysFromFiles(
#             path=instructions.path_dir_input,
#             slicing=instructions.slicing,
#         )

#     def compute(self):
#         instructions = self.instructions
#         serie = self.serie_arrays

#         slicing_tuples = serie.get_slicing_tuples()

#         # create output name
#         strindices_first_file = serie.compute_str_indices_from_indices(
#             *[indices[0] for indices in slicing_tuples]
#         )
#         strindices_last_file = serie.compute_str_indices_from_indices(
#             *[indices[1] - 1 for indices in slicing_tuples]
#         )
#         name_file = (
#             serie.base_name
#             + serie.get_separator_base_index()
#             + strindices_first_file
#             + "-"
#             + strindices_last_file
#             + "."
#             + serie.extension_file
#         )
#         path_save = os.path.join(instructions.path_dir_output, name_file)

#         # compute the average
#         logger.info("Compute the average")
#         a = serie.get_array_from_index(0)
#         mean = np.zeros_like(a, dtype=np.float32)
#         nb_fields = 0
#         for a in serie.iter_arrays():
#             mean += a
#             nb_fields += 1
#         mean /= nb_fields

#         logger.info("Save in file:\n%s", path_save)
#         scipy.misc.imsave(path_save, mean)
#         return mean


def params_piv_from_uvmat_xml(instructions, args):
    params = TopologyPIV.create_default_params()

    params.series.path = instructions.path_file_input

    params.series.str_subset = instructions.str_subset
    params.series.ind_stop = instructions.ind_stop
    params.series.ind_start = instructions.ind_start

    params.saving.path = instructions.path_dir_output

    n0 = int(instructions.action_input.civ1.search_box_size.split("\t")[0])
    if n0 % 2 == 1:
        n0 -= 1

    params.piv0.shape_crop_im0 = n0

    if hasattr(instructions.action_input, "patch2"):
        params.multipass.subdom_size = (
            instructions.action_input.patch2.sub_domain_size
        )
    else:
        params.multipass.use_tps = False

    if hasattr(instructions.action_input, "civ2"):
        params.multipass.number = 2

    modif_fluidimage_params(params, args)

    return params


class ActionPIVFromUvmatXML(ActionBase):
    """Compute the average and save as a png file."""

    def __init__(self, instructions, args):
        self.instructions = instructions
        self.params = params_piv_from_uvmat_xml(instructions, args)
        logger.info("Initialize Fluidimage computations with parameters:")
        logger.info(self.params._make_xml_text())
        self.topology = TopologyPIV(self.params)

    def compute(self):
        t = time()
        self.topology.compute()
        t = time() - t
        print(f"elapsed time: {t} s")


# actions_classes = {"aver_stat": ActionAverage, "civ_series": ActionPIVFromUvmatXML}
actions_classes = {"civ_series": ActionPIVFromUvmatXML}

programs = ["uvmat", "fluidimage"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Path file.", type=str, nargs="?")
    parser.add_argument(
        "-m",
        "--mode",
        help="'ask', 'new_dir', 'complete' or 'recompute'.",
        type=str,
        default=None,
    )

    return parser.parse_args()


def modif_fluidimage_params(params, args):
    if args.mode is not None:
        try:
            params.saving.how = args.mode
        except AttributeError:
            pass


def main():
    reset_logger()
    config_logging("info")

    args = parse_args()

    path_instructions_xml = args.path

    params = ParamContainer(path_file=path_instructions_xml)

    program = None
    try:
        params.Action.ActionName
    except AttributeError:
        try:
            program = params._value_text["program"]
        except (AttributeError, KeyError):
            pass
    else:
        program = "uvmat"

    if program not in programs:
        raise ValueError("Can not detect the program to launch.")

    logger.info(
        "\n%s: using instructions in xml file:\n%s",
        time_as_str(2),
        path_instructions_xml,
    )

    kwargs_compute = {}

    if program == "uvmat":
        tidy_uvmat_instructions(params)
        action_name = params.action.action_name
        logger.info(
            'Check if the action "%s" is implemented by FluidImage', action_name
        )
        if action_name not in actions_classes.keys():
            raise NotImplementedError(
                'action "' + action_name + '" is not yet implemented.'
            )

        cls = actions_classes[action_name]
        action = cls(params, args)

    elif program == "fluidimage":
        cls = import_class(
            params._value_text["module"], params._value_text["class"]
        )
        modif_fluidimage_params(params, args)
        action = cls(params)

        try:
            compute_kwargs = params.compute_kwargs
        except AttributeError:
            pass
        else:
            kwargs_compute = compute_kwargs._make_dict_tree()

    return action.compute(**kwargs_compute)


if __name__ == "__main__":
    result = main()
