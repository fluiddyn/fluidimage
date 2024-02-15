"""
Launchers for topologies used for GUI (:mod:`fluidimage.topologies.launcher`)
=============================================================================

.. autoclass:: TopologyLauncher
   :members:
   :private-members:

.. autoclass:: TopologyPIVLauncher
   :members:
   :private-members:

.. autoclass:: TopologyPreprocLauncher
   :members:
   :private-members:

"""
import inspect
import json

from fluidimage.topologies.piv import TopologyPIV
from fluidimage.topologies.preproc import TopologyPreproc


def _get_args_bound_method(method):
    try:  # python 3
        arg_spec = inspect.getfullargspec(method)
    except AttributeError:  # python 2
        arg_spec = inspect.getargspec(method.__func__)

    # key args without 'self' and 'params'
    key_args = arg_spec.args[2:]
    defaults = arg_spec.defaults[1:]
    return key_args, defaults


class TopologyLauncher:
    """Launcher (class to be subclassed)."""

    Topology = TopologyPIV

    @classmethod
    def create_default_params(cls):
        params = cls.Topology.create_default_params()

        key_args, defaults = _get_args_bound_method(cls.Topology.__init__)

        params._set_child("topology")

        for key, value in zip(key_args, defaults):
            params.topology._set_attrib(key, value)

        params.topology._set_doc(cls.Topology.__doc__)

        key_args, defaults = _get_args_bound_method(cls.Topology.compute)
        params.topology._set_child("compute")

        for key, value in zip(key_args, defaults):
            params.topology.compute._set_attrib(key, value)

        params.topology.compute._set_doc(cls.Topology.compute.__doc__)

        params._set_internal_attr(
            "_value_text",
            json.dumps(
                {
                    "program": "fluidimage",
                    "module": cls.__module__,
                    "class": cls.__name__,
                }
            ),
        )

        return params

    def __init__(self, params):
        self.params = params
        kwargs = params.topology._make_dict_attribs()
        print(kwargs)
        self.topology = self.Topology(params, **kwargs)

    def compute(self):
        kwargs = self.params.topology.compute._make_dict_attribs()
        print(kwargs)
        self.topology.compute(**kwargs)


class TopologyPIVLauncher(TopologyLauncher):
    """PIV topology launcher"""


class TopologyPreprocLauncher(TopologyLauncher):
    """Preproc topology launcher"""

    Topology = TopologyPreproc
