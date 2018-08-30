"""Import OpenCV
================
Convenience module to import OpenCV and warn if the package was not found.
Hint: install it!

"""

try:
    import cv2
except ModuleNotFoundError:
    print(
        "Warning: OpenCV must be built and installed with python bindings "
        "to use this module. Install using either of the following\n"
        "    pip install opencv-python\n"
        "    conda install -c conda-forge opencv"
    )
    cv2 = None
