"""Topology for optical flow (:mod:`fluidimage.topologies.optical_flow`)
========================================================================

.. autoclass:: TopologyOpticalFlow
   :members:
   :private-members:

"""

import sys

from fluidimage.works.optical_flow import WorkOpticalFlow

from .piv import TopologyPIV


class TopologyOpticalFlow(TopologyPIV):
    """Optical flow topology (Lukas Kanade method)"""

    WorkVelocimetry = WorkOpticalFlow
    _short_name = "optflow"

    @classmethod
    def create_default_params(cls):
        """Class method returning the default parameters.

        Typical usage::

          params = TopologyOpticalFlow.create_default_params()
          # modify parameters here
          ...

          topo = TopologyOpticalFlow(params)

        """

        params = super().create_default_params()
        return params


Topology = TopologyOpticalFlow

if "sphinx" in sys.modules:
    _params = Topology.create_default_params()
    __doc__ += _params._get_formatted_docs()
