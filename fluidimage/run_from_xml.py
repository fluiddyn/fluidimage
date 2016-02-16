"""Uvmat interface (:mod:`fluidimage.run_from_xml`)
===================================================

From matlab, something like::

  system(['python -m fluidimage.run_from_xml ' path_to_instructions_xml])

"""

import os
import sys

import numpy as np
import scipy

import logging
log_level = logging.INFO  # to get information messages
# log_level = logging.WARNING  # no information messages
logging.basicConfig(format='%(message)s',
                    level=log_level)

from fluiddyn.util.paramcontainer import ParamContainer, tidy_container
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles


class InstructionsUVMAT(ParamContainer):
    """
    instructions = InstructionsUVMAT(path_file=(
        '../image_samples/Oseen/Images.civ_uvmat/0_XML/Oseen_center_1.xml'))


    """

    def __init__(self, **kargs):

        if 'tag' not in kargs:
            kargs['tag'] = 'instructions_uvmat'

        super(InstructionsUVMAT, self).__init__(**kargs)

        if kargs['tag'] == 'instructions_uvmat' and 'path_file' in kargs:
            self._init_root()

    def _init_root(self):

        # get nicer names and a simpler organization...
        tidy_container(self)

        input_table = self.input_table.split(' & ')
        self.input_table = '|'.join(input_table)
        name_file = ''.join(input_table[2:])
        path_dir_root = input_table[0]
        path_dir_input = os.path.join(path_dir_root, input_table[1])
        path_file_input = os.path.join(path_dir_input, name_file)
        self._set_attrib('path_dir_input', path_dir_input)
        self._set_attrib('path_file_input', path_file_input)
        
        self._set_attrib(
            'path_dir_output',
            path_dir_input + self.output_dir_ext)

        ir = self.index_range
        slice0 = [ir.first_i, ir.last_i+1, ir.incr_i]
        try:
            slice1 = [ir.first_j-1, ir.last_j, ir.incr_j]
            self._set_attrib('index_slices', [slice0, slice1])
        except AttributeError:
            self._set_attrib('index_slices', [slice0])


class ActionBase(object):
    def __init__(self, instructions):
        self.instructions = instructions

        # create the serie of arrays
        logging.info('Create the serie of arrays')
        self.serie_arrays = SerieOfArraysFromFiles(
            path=instructions.path_dir_input,
            index_slices=instructions.index_slices)


class ActionAverage(ActionBase):
    """Compute the average and save as a png file."""
    def run(self):
        instructions = self.instructions
        serie = self.serie_arrays
        # compute the average
        logging.info('Compute the average')
        a = serie.get_array_from_index(0)
        mean = np.zeros_like(a, dtype=np.float32)
        nb_fields = 0
        for a in serie.iter_arrays():
            mean += a
            nb_fields += 1
        mean /= nb_fields

        strindices_first_file = serie._compute_strindices_from_indices(
            *[indices[0] for indices in instructions.index_slices])
        strindices_last_file = serie._compute_strindices_from_indices(
            *[indices[1]-1 for indices in instructions.index_slices])

        name_file = (serie.base_name + serie._separator_base_index
                     + strindices_first_file + '-' + strindices_last_file
                     + '.' + serie.extension_file)

        path_save = os.path.join(instructions.path_dir_output, name_file)
        logging.info('Save in file:\n%s',  path_save)
        scipy.misc.imsave(path_save, mean)
        return mean


actions_classes = {'aver_stat': ActionAverage}


def main():
    if len(sys.argv) > 1:
        path_instructions_xml = sys.argv[1]
    else:
        raise ValueError

    logging.info('\nFrom Python, start with instructions in xml file:\n%s',
                 path_instructions_xml)

    instructions = InstructionsUVMAT(path_file=path_instructions_xml)

    action_name = instructions.action.action_name
    logging.info('Check if the action "%s" is implemented by FluidImage',
                 action_name)
    if action_name not in actions_classes.keys():
        raise NotImplementedError(
            'action "' + action_name + '" is not yet implemented.')

    Action = actions_classes[action_name]
    action = Action(instructions)
    return action.run()


if __name__ == '__main__':

    result = main()
