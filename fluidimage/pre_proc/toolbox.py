"""Preprocess toolbox (:mod:`fluidimage.pre_proc.toolbox`)
==========================================================
A toolbox for preprocessing images.

.. currentmodule:: fluidimage.pre_proc.toolbox

Provides:

.. autoclass:: PreprocTools
   :members:
   :private-members:

"""

import inspect
from ..util.util import logger


class PreprocTools(object):
    """Wrapper class for functions in the current module."""

    @classmethod
    def _complete_class_with_tools(cls, params):
        """
        Dynamically add the functions as staticmethods of the present class.
        Also create default parameters by inspecting the function argument list.

        """
        params = params.preproc.tools
        if params.backend == 'python':
            from . import _toolbox_py as tools
        elif params.backend == 'opencv':
            from . import _toolbox_cv as tools
        else:
            raise ImportError('Unknown backend: %s' % params.backend)

        params._set_attribs({'available_tools': tools.__all__,
                             'sequence': None})

        for tool in params.available_tools:
            func = tools.__dict__[tool]

            # Add tools as `staticmethods` of the class
            setattr(PreprocTools, tool, func)

            # TODO: Replace with inspect.getfullargspec (Python >= 3).
            func_args = inspect.getcallargs(func)
            for arg in func_args.keys():
                if arg in ['img']:
                    # Remove arguments which are not parameters
                    del(func_args[arg])

            func_args.update({'enable': False})

            # New parameter child for each tool and parameter attributes
            # from its function arguments and default values
            params._set_child(tool, attribs=func_args)

            # Adds docstring to the parameter
            if func.func_doc is not None:
                enable_doc = 'enable : bool\n' + \
                             '        Set as `True` to enable the tool'
                params.__dict__[tool]._set_doc(func.func_doc + enable_doc)

    def __init__(self, params):
        self.params = params.preproc.tools

    def __call__(self, img):
        """
        Apply all preprocessing tools for which `enable` is `True`.
        Return the preprocessed image (numpy array).

        Parameters
        ----------
        img : array_like
            Single image as numpy array or multiple images as array-like object

        """
        sequence = self.params.sequence
        if sequence is None:
            sequence = self.params.available_tools

        for tool in sequence:
            tool_params = self.params.__dict__[tool]
            if tool_params.enable:
                logger.debug('Apply ' + tool)
                kwargs = tool_params._make_dict()
                for k in kwargs.keys():
                    if k in ['_attribs', '_tag', '_tag_children', 'enable']:
                        kwargs.pop(k)

                cls = self.__class__
                img = cls.__dict__[tool](img, **kwargs)

        return img
