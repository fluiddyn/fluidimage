"""Import OpenCV
================
Convenience module to import OpenCV and warn if the package was not found.
Hint: install it!

"""

from warnings import warn

from fluiddyn.util.opencv import cv2, error_import_cv2

if isinstance(error_import_cv2, ModuleNotFoundError):
    warn(
        "OpenCV must be built and installed with python bindings "
        "to use this module. Install using either of the following\n"
        "    pip install opencv-python\n"
        "    conda install -c conda-forge opencv"
    )

__all__ = ["cv2"]
