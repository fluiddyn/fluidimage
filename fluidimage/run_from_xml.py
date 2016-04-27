"""Uvmat interface (:mod:`fluidimage.run_from_xml`)
===================================================

From matlab, something like::

  system(['python -m fluidimage.run_from_xml ' path_to_instructions_xml])


.. todo::

   UVmat civserie: input and output.

"""

import os
import sys
import logging
from time import time

import numpy as np
import scipy

from . import (
    config_logging, ParamContainer, SerieOfArraysFromFiles)

from fluiddyn.util.paramcontainer import tidy_container

from fluidimage.topologies.piv import TopologyPIV


config_logging('info')

logger = logging.getLogger('fluidimage')


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
        path_file_input = os.path.abspath(self.input_table.replace('|', ''))
        path_dir_input = os.path.dirname(path_file_input)
        self._set_attrib('path_dir_input', path_dir_input)
        self._set_attrib('path_file_input', path_file_input)

        self._set_attrib(
            'path_dir_output',
            path_dir_input + self.output_dir_ext)

        ir = self.index_range
        slice0 = [ir.first_i, ir.last_i+1, ir.incr_i]
        try:
            slice1 = [ir.first_j-1, ir.last_j, ir.incr_j]
            slices = [slice0, slice1]
        except AttributeError:
            slices = [slice0]

        self._set_attrib('index_slices', slices)

        if len(slices) == 1:
            strcouple = '{}+i: {}+i'.format(ir.first_i, ir.first_i+ir.incr_i+1)
            ind_stop = 1 + ir.last_i - ir.first_i
        else:
            raise NotImplementedError
        self._set_attrib('strcouple', strcouple)
        self._set_attrib('ind_stop', ind_stop)


class ActionBase(object):
    def __init__(self, instructions):
        self.instructions = instructions

        # create the serie of arrays
        logger.info('Create the serie of arrays')
        self.serie_arrays = SerieOfArraysFromFiles(
            path=instructions.path_dir_input,
            index_slices=instructions.index_slices)


class ActionAverage(ActionBase):
    """Compute the average and save as a png file."""
    def run(self):
        instructions = self.instructions
        serie = self.serie_arrays
        # compute the average
        logger.info('Compute the average')
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

        name_file = (serie.base_name + serie._separator_base_index +
                     strindices_first_file + '-' + strindices_last_file +
                     '.' + serie.extension_file)

        path_save = os.path.join(instructions.path_dir_output, name_file)
        logger.info('Save in file:\n%s',  path_save)
        scipy.misc.imsave(path_save, mean)
        return mean


def params_from_uvmat_xml(instructions):
    params = TopologyPIV.create_default_params()

    params.series.path = instructions.input_table.replace('|', '')

    params.series.strcouple = instructions.strcouple
    params.series.ind_stop = instructions.ind_stop

    params.saving.path = instructions.path_dir_output

    n0 = int(instructions.action_input.civ1.search_box_size.split('\t')[0])
    if n0 % 2 == 1:
        n0 -= 1

    params.piv0.shape_crop_im0 = n0

    params.multipass.subdom_size = \
        instructions.action_input.patch2.sub_domain_size

    if hasattr(instructions.action_input, 'civ2'):
        params.multipass.number = 2

    return params


class ActionPIV(ActionBase):
    """Compute the average and save as a png file."""
    def __init__(self, instructions):
        self.instructions = instructions
        self.params = params_from_uvmat_xml(instructions)
        self.topology = TopologyPIV(self.params)

    def run(self):
        t = time()
        self.topology.compute(sequential=False)
        t = time() - t

        print('ellapsed time: {}s'.format(t))

actions_classes = {'aver_stat': ActionAverage,
                   'civ_series': ActionPIV}


def main():
    if len(sys.argv) > 1:
        path_instructions_xml = sys.argv[1]
    else:
        raise ValueError

    logger.info('\nFrom Python, start with instructions in xml file:\n%s',
                path_instructions_xml)

    instructions = InstructionsUVMAT(path_file=path_instructions_xml)

    action_name = instructions.action.action_name
    logger.info('Check if the action "%s" is implemented by FluidImage',
                action_name)
    if action_name not in actions_classes.keys():
        raise NotImplementedError(
            'action "' + action_name + '" is not yet implemented.')

    Action = actions_classes[action_name]
    action = Action(instructions)
    return action.run()


if __name__ == '__main__':

    result = main()
