"""Preprocess toolbox (:mod:`fluidimage.preproc.toolbox`)
=========================================================

A toolbox of filters which operate on a single image (numpy array).
cf. http://www.scipy-lectures.org/advanced/image_processing/

Provides:

.. autoclass:: PreprocToolsBase
   :members:

.. autoclass:: PreprocToolsPy
   :members:

.. autoclass:: PreprocToolsCV
   :members:

"""

import inspect
from ..util.util import logger


class PreprocToolsBase(object):
    """Base class for wrapping preprocessing functions into a class."""

    @classmethod
    def _get_backend(cls):
        raise NotImplementedError

    @classmethod
    def create_default_params(cls, params):
        """Create default parameters from the function argument list.

        """
        tools = cls._get_backend()
        available_tools = tools.__all__

        params.preproc._set_child("tools")
        params = params.preproc.tools
        params._set_attribs({"available_tools": tools.__all__, "sequence": None})

        for tool in available_tools:
            func = tools.__dict__[tool]

            # TODO: Replace with inspect.getfullargspec (Python >= 3).
            func_args = inspect.getcallargs(func)
            for arg in list(func_args.keys()):
                if arg in ["img"]:
                    # Remove arguments which are not parameters
                    del (func_args[arg])

            func_args.update({"enable": False})

            # New parameter child for each tool and parameter attributes
            # from its function arguments and default values
            params._set_child(tool, attribs=func_args)

            # Adds docstring to the parameter
            try:
                func_doc = func.func_doc
            except AttributeError:
                func_doc = func.__doc__

            if func_doc is not None:
                enable_doc = (
                    "enable : bool\n" + "        Set as `True` to enable the tool"
                )
                params[tool]._set_doc(func_doc + enable_doc)

    @classmethod
    def _complete_class_with_tools(cls):
        """
        Dynamically add the global functions in this module as staticmethods of
        the present class.

        """
        tools = cls._get_backend()
        available_tools = tools.__all__

        for tool in available_tools:
            func = tools.__dict__[tool]

            # Add tools as `staticmethods` of the class
            setattr(cls, tool, func)

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
            tool_params = self.params[tool]
            if tool_params.enable:
                logger.debug("Apply " + tool)
                kwargs = tool_params._make_dict_attribs()
                for k in list(kwargs.keys()):
                    if k == "enable":
                        kwargs.pop(k)

                cls = self.__class__
                img = cls.__dict__[tool](img, **kwargs)

        return img


class PreprocToolsPy(PreprocToolsBase):
    """Wrapper class for functions in _toolbox_py module."""

    @classmethod
    def _get_backend(cls):
        from . import _toolbox_py as tools

        return tools


PreprocToolsPy._complete_class_with_tools()


class PreprocToolsCV(PreprocToolsBase):
    """Wrapper class for functions in _toolbox_cv module."""

    @classmethod
    def _get_backend(cls):
        from . import _toolbox_cv as tools

        return tools


PreprocToolsCV._complete_class_with_tools()
