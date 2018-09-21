"""
ParamList (:mod:`fluidimage.util.paramlist`)
============================================

To launch topologies in bulk, parameters have to be stored in bulk.

Provides inherited list designed to store `ParamContainer` instances
for a directory tree. It assumes the following hierarchy of directories:

.. Experiment --> Camera --> Level --> Images

Provides:

.. autoclass:: ParamListBase
   :members:
   :private-members:

.. autoclass:: ParamListPreproc
   :members:
   :private-members:

"""

import os
import string
from copy import deepcopy
from glob import glob
from warnings import warn
from fluidimage.topologies.preproc import TopologyPreproc
from fluidimage.topologies.piv import TopologyPIV


class ParamListBase(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.TopologyClass = None

        # Dictionary of functions which return parameters
        self.camera_specific_params = kwargs["camera_specific_params"]
        self.path_list = kwargs["path_list"]

    def init_directory(self, ind_exp, camera):
        self.path = self.path_list[ind_exp]
        if camera not in self.camera_specific_params.keys():
            raise ValueError(
                "Unexpected camera name %s Expected %s"
                % (camera, self.camera_specific_params.keys())
            )

        self.camera = camera
        self.frames = self._detect_frames()

    def _detect_frames(self):
        """Detects if it is single (series) or double (burst) frame expt."""

        if "PCO" in self.camera:
            pattern = os.path.join(self.path, "*frame*piv*")
            scan_piv_file = glob(pattern)[0]
            if "single_frame" in scan_piv_file:
                return 1

            elif "double_frame" in scan_piv_file:
                return 2

            else:
                raise ValueError("Not sure if it is single/double frame expt.")

        elif "Dalsa" in self.camera:
            warn("Assuming files to be single frame in Dalsa")
            return 1

    def _set_complete_path(self, params, level):
        raise NotImplementedError

    def _get_complete_path(self, params):
        raise NotImplementedError

    def get_levels(self, pattern):
        """Returns a the list of subdirectories under a particular experiment,
        and a camera.

        """
        if pattern is None:
            pattern = "level??"

        pattern = os.path.join(self.path, self.camera, pattern)
        levels = [os.path.basename(subdir) for subdir in glob(pattern)]
        return levels

    def fill_params(self, level, **kwargs):
        """
        Initialize parameters for a particular experiment, camera and level.

        """
        params = self.TopologyClass.create_default_params(**kwargs)
        params = self._set_complete_path(params, level)

        get_params = self.camera_specific_params[self.camera]

        if self.frames == 1:
            params = get_params(params, self.frames)
            self.append(params)
        elif self.frames > 1:
            for i in range(self.frames):
                letter = string.lowercase[i]
                params_copy = deepcopy(params)
                params_copy = get_params(params_copy, self.frames, letter)
                self.append(params_copy)

    def launch_topologies(self, seq=False, verbose=0):
        for params in self:
            if verbose is None:
                log = None
            if verbose >= 1:
                print(self._get_complete_path(params))
                if verbose == 1:
                    log = "info"
                else:
                    log = "debug"

            topology = self.TopologyClass(params, logging_level=log)
            topology.compute(sequential=seq)


class ParamListPreproc(ParamListBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TopologyClass = TopologyPreproc

    def _set_complete_path(self, params, level):
        params.preproc.series.path = os.path.join(self.path, self.camera, level)
        return params

    def _get_complete_path(self, params):
        return params.preproc.series.path


class ParamListPIV(ParamListBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TopologyClass = TopologyPIV

    def _set_complete_path(self, params, level):
        params.series.path = os.path.join(self.path, self.camera, level)
        return params

    def _get_complete_path(self, params):
        return params.series.path
