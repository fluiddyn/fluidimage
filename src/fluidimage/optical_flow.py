"""Helper module for optical flow computations

Provides

.. autoclass:: Work
   :members:
   :private-members:

.. autoclass:: Topology
   :members:
   :private-members:

"""

from .topologies.optical_flow import TopologyOpticalFlow
from .works.optical_flow import WorkOpticalFlow

Work = WorkOpticalFlow
Topology = TopologyOpticalFlow
