"""Uvmat interface (:mod:`fluidimage.run_from_xml`)
===================================================

From matlab, something like::

  system(['python -m fluidimage.run_from_xml ' path_to_instructions_xml])


.. todo::

   UVmat civserie: input and output.

"""

import argparse
import os
from time import time

import numpy as np
import scipy

from fluiddyn.util import import_class
from fluiddyn.util.paramcontainer import tidy_container
from fluidimage.topologies.piv import TopologyPIV

from . import (
    ParamContainer,
    SerieOfArraysFromFiles,
    config_logging,
    logger,
    reset_logger,
)


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
        strcouple = f"i:i+{ir.incr_i + 1}:{ir.incr_i}"
        ind_start = ir.first_i

        if ir.last_i is None:
            ind_stop = None
        else:
            ind_stop = ir.last_i - ir.incr_i + 1
    else:
        raise NotImplementedError

    params._set_attrib("strcouple", strcouple)
    params._set_attrib("ind_stop", ind_stop)
    params._set_attrib("ind_start", ind_start)


class ActionBase:
    def __init__(self, instructions, args):
        self.instructions = instructions

        # create the serie of arrays
        logger.info("Create the serie of arrays")
        self.serie_arrays = SerieOfArraysFromFiles(
            path=instructions.path_dir_input,
            index_slices=instructions.index_slices,
        )


class ActionAverage(ActionBase):
    """Compute the average and save as a png file."""

    def compute(self):
        instructions = self.instructions
        serie = self.serie_arrays
        # compute the average
        logger.info("Compute the average")
        a = serie.get_array_from_index(0)
        mean = np.zeros_like(a, dtype=np.float32)
        nb_fields = 0
        for a in serie.iter_arrays():
            mean += a
            nb_fields += 1
        mean /= nb_fields

        strindices_first_file = serie._compute_strindices_from_indices(
            *[indices[0] for indices in instructions.index_slices]
        )
        strindices_last_file = serie._compute_strindices_from_indices(
            *[indices[1] - 1 for indices in instructions.index_slices]
        )

        name_file = (
            serie.base_name
            + serie._separator_base_index
            + strindices_first_file
            + "-"
            + strindices_last_file
            + "."
            + serie.extension_file
        )

        path_save = os.path.join(instructions.path_dir_output, name_file)
        logger.info("Save in file:\n%s", path_save)
        scipy.misc.imsave(path_save, mean)
        return mean


def params_piv_from_uvmat_xml(instructions, args):
    params = TopologyPIV.create_default_params()

    params.series.path = instructions.path_file_input

    params.series.strcouple = instructions.strcouple
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


class ActionPIV(ActionBase):
    """Compute the average and save as a png file."""

    def __init__(self, instructions, args):
        self.instructions = instructions
        self.params = params_piv_from_uvmat_xml(instructions, args)
        logger.info("Initialize fluidimage computations with parameters:")
        logger.info(self.params._make_xml_text())
        self.topology = TopologyPIV(self.params)

    def compute(self):
        t = time()
        self.topology.compute(sequential=False)
        t = time() - t
        print(f"ellapsed time: {t}s")


actions_classes = {"aver_stat": ActionAverage, "civ_series": ActionPIV}

programs = ["uvmat", "fluidimage"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Path file.", type=str, nargs="?")
    parser.add_argument(
        "-m",
        "--mode",
        help="‘ask’, ‘new_dir’, ‘complete’ or ‘recompute’.",
        type=str,
        default="ask",
    )

    return parser.parse_args()


def modif_fluidimage_params(params, args):
    try:
        params.saving.how = args.mode
    except AttributeError:
        pass


def main():

    reset_logger()
    config_logging("info")

    args = parse_args()

    path_instructions_xml = args.path
    logger.info(
        "\nFrom Python, start with instructions in xml file:\n%s",
        path_instructions_xml,
    )

    params = ParamContainer(path_file=path_instructions_xml)

    program = None

    try:
        params.Action.ActionName
    except AttributeError:
        pass
    else:
        program = "uvmat"

    try:
        program = params._value_text["program"]
    except (AttributeError, KeyError):
        pass

    if program not in programs:
        raise ValueError("Can not detect the program to launch.")

    print(args)

    print(f"using input file from {program}")

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

    return action.compute()


if __name__ == "__main__":

    result = main()
