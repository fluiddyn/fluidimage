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
        params.saving.postfix = "optflow"
        return params


Topology = TopologyOpticalFlow

if "sphinx" in sys.modules:
    params = Topology.create_default_params()
    __doc__ += params._get_formatted_docs()


if __name__ == "__main__":
    from fluidimage import path_image_samples

    params = Topology.create_default_params()

    params.series.path = str(path_image_samples / "Karman/Images")
    params.series.ind_start = 1
    params.series.ind_step = 2

    params.mask.strcrop = ":, 50:500"

    # params.preproc.im2im = "numpy.ones_like"

    # params.saving.how = 'complete'
    params.saving.postfix = "optical_flow_example"

    topo = Topology(params, logging_level="info")

    # topo.make_code_graphviz("tmp.dot")
    topo.compute()
