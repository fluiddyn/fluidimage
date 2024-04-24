"""Compute with Fluidimage from a xml file

It is used in particular as a Uvmat interface. From Matlab, something like::

  system(['python -m fluidimage.run_from_xml ' path_to_instructions_xml])

"""

import argparse

from fluiddyn import time_as_str
from fluiddyn.util import import_class
from fluidimage.uvmat import actions_classes, tidy_uvmat_instructions

from . import ParamContainer, config_logging, logger, reset_logger

supported_programs = ["uvmat", "fluidimage"]


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

    if program not in supported_programs:
        raise ValueError("Can not detect the program to launch.")

    logger.info(
        "\n%s: using instructions in xml file:\n%s",
        time_as_str(2),
        path_instructions_xml,
    )

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
        params = cls.params_from_uvmat_xml(params)

    elif program == "fluidimage":
        cls = import_class(
            params._value_text["module"], params._value_text["class"]
        )

    modif_fluidimage_params(params, args)
    action = cls(params)

    try:
        compute_kwargs = params.compute_kwargs
    except AttributeError:
        kwargs_compute = {}
    else:
        kwargs_compute = compute_kwargs._make_dict_tree()

    action.compute(**kwargs_compute)
    return action
