"""UVmat interface

"""

import os
from abc import ABC, abstractmethod
from logging import warning
from time import time

from fluiddyn.util.paramcontainer import tidy_container
from fluidimage import logger
from fluidimage.topologies.piv import TopologyPIV


def tidy_uvmat_instructions(instructions):
    """Clean up UVmat instructions with nicer names and a simpler organization"""
    instructions._set_internal_attr("_tag", "instructions_uvmat")

    tidy_container(instructions)

    instructions.input_table = input_table = instructions.input_table.split(" & ")
    for i in range(1, len(input_table)):
        if input_table[i].startswith("/"):
            input_table[i] = input_table[i][1:]

    filename = "".join(input_table[2:])
    path_dir = os.path.join(*input_table[:2])
    path_file_input = os.path.abspath(os.path.join(path_dir, filename))
    path_dir_input = path_dir
    instructions._set_attrib("path_dir_input", path_dir_input)
    instructions._set_attrib("path_file_input", path_file_input)
    instructions._set_attrib(
        "path_dir_output", path_dir_input + instructions.output_dir_ext
    )


class ActionBase(ABC):
    name: str
    computer_cls: type

    @classmethod
    @abstractmethod
    def params_from_uvmat_xml(cls, instructions):
        """Create fluidimage params from UVmat xml"""

    @classmethod
    def set_params_series(cls, params, instructions):
        """Set params.series from UVmat xml"""
        ir = instructions.index_range
        if not hasattr(ir, "incr_i"):
            ir._set_attrib("incr_i", 1)

        if not hasattr(ir, "first_i"):
            warning("Warning: no attribute first_i in xml UVmat file.")
            ir._set_attrib("first_i", 1)

        if not hasattr(ir, "last_i"):
            warning("Warning: no attribute last_i in xml UVmat file.")
            ir._set_attrib("last_i", None)

        if not hasattr(ir, "first_j"):
            str_subset = "i:i+2:1"
            ind_start = ir.first_i
            ind_step = ir.incr_i

            if ir.last_i is None:
                ind_stop = None
            else:
                ind_stop = ir.last_i + 1
        else:
            raise NotImplementedError

        params.series.path = instructions.path_file_input
        params.series.str_subset = str_subset
        params.series.ind_start = ind_start
        params.series.ind_step = ind_step
        params.series.ind_stop = ind_stop

    def __init__(self, params):
        """Initialize the action class"""
        self.params = params
        logger.info("Initialize Fluidimage computations with parameters:")
        logger.info(self.params._make_xml_text())
        self.computer = self.computer_cls(self.params)

    def compute(self):
        """Perform the computation"""
        t = time()
        self.computer.compute()
        t = time() - t
        print(f"elapsed time: {t} s")


# class ActionAverage(ActionBase):
#     """Compute the average and save as a png file."""

#     def __init__(self, instructions):
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


class ActionPIVFromUvmatXML(ActionBase):
    """Compute the average and save as a png file."""

    name = "civ_series"
    computer_cls = TopologyPIV

    @classmethod
    def params_from_uvmat_xml(cls, instructions):
        params = cls.computer_cls.create_default_params()
        cls.set_params_series(params, instructions)

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

        return params


actions_classes = {
    cls.name: cls
    for cls in locals().values()
    if isinstance(cls, type)
    and issubclass(cls, ActionBase)
    and cls is not ActionBase
}
