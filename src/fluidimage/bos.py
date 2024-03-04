"""Helper module for BOS computations

Provides

.. autoclass:: Work
   :members:
   :private-members:

.. autoclass:: Topology
   :members:
   :private-members:

"""

from .topologies.bos import TopologyBOS
from .works.bos import WorkBOS

Work = WorkBOS
Topology = TopologyBOS
