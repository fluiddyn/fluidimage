"""Works optical flow (:mod:`fluidimage.works.optical_flow`)
============================================================

See
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade
(and https://github.com/groussea/opyflow by Gauthier Rousseau)

"""

import cv2
import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage.data_objects.piv import ArrayCouple, HeavyPIVResults

from .with_mask import BaseWorkWithMask


def dict_from_params(params):
    return {k: v for k, v in params.__dict__.items() if not k.startswith("_")}


def optical_flow(
    im0,
    im1,
    feature_params,
    lk_params,
    threshold_diff_ab_ba=10,
    vmin=0,
    vmax=np.inf,
):

    positions0 = cv2.goodFeaturesToTrack(im0, **feature_params)

    positions1, st, err = cv2.calcOpticalFlowPyrLK(
        im0, im1, positions0, None, **lk_params
    )
    positions0r, st, err = cv2.calcOpticalFlowPyrLK(
        im1, im0, positions1, None, **lk_params
    )

    positions = positions0.reshape(-1, 2)
    displacements = positions1.reshape(-1, 2) - positions

    diff = abs(positions0 - positions0r).reshape(-1, 2).max(-1)

    correct_values = diff < threshold_diff_ab_ba
    positions = positions[correct_values]
    displacements = displacements[correct_values]

    if vmin is not None or vmax is not None:
        if vmin is None:
            vmin = 0

        if vmax is None:
            vmax = np.inf

        norm = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
        cond = (vmin < norm) & (norm < vmax)
        positions = positions[cond]
        displacements = displacements[cond]

    return positions, displacements


class WorkOpticalFlow(BaseWorkWithMask):
    @classmethod
    def create_default_params(cls):
        "Create an object containing the default parameters (class method)."
        params = ParamContainer(tag="params")
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params):
        params._set_child(
            "optical_flow",
            attribs=dict(
                winSize=(32, 32),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    50,
                    0.03,
                ),
            ),
        )
        params.optical_flow._set_doc(
            """Parameters for the flow calculation using Lukas Kanade method

See https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html

The algorithm is "pyramidal" (multiple passes).

winSize :

  Size of the search window at each pyramid level.

maxLevel :

  0-based maximal pyramid level number; if set to 0, pyramids are not used
  (single level), if set to 1, two levels are used, and so on; if pyramids are
  passed to input then algorithm will use as many levels as pyramids have but no
  more than maxLevel.

criteria :

  Termination criteria of the iterative search algorithm (after the specified
  maximum number of iterations criteria.maxCount or when the search window
  moves by less than criteria.epsilon.

"""
        )

        params._set_child(
            "features",
            attribs=dict(
                maxCorners=70000, qualityLevel=0.09, minDistance=4, blockSize=16
            ),
        )
        params.features._set_doc(
            """Parameters for the Good Feature to Track algorithm (Shi-Tomasi Corner Detector)

See https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack

maxCorners :

  Maximum number of corners to return. If there are more corners than are found,
  the strongest of them is returned.

qualityLevel :

  Parameter characterizing the minimal accepted quality of image corners. The
  parameter value is multiplied by the best corner quality measure, which is the
  minimal eigenvalue or the Harris function response. The corners with the
  quality measure less than the product are rejected. For example, if the best
  corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the
  corners with the quality measure less than 15 are rejected.

minDistance :

  Minimum possible Euclidean distance between the returned corners.

blockSize :

  Size of an average block for computing a derivative covariation matrix over
  each pixel neighborhood."""
        )

        cls._complete_params_with_default_mask(params)

        params._set_child(
            "filters",
            attribs={
                "threshold_diff_ab_ba": 1.0,
                "displacement_min": None,
                "displacement_max": None,
            },
        )

        params.filters._set_doc(
            """
Parameters indicating how are detected and processed false vectors.

threshold_diff_ab_ba : 1.

    ???

displacement_min : None

    Vectors smaller than `displacement_min` are considered as false vectors.

displacement_max : None

    Vectors larger than `displacement_max` are considered as false vectors.

"""
        )

    def __init__(self, params):

        self.params = params
        self.dict_params_features = dict_from_params(self.params.features)
        self.dict_params_flow = dict_from_params(self.params.optical_flow)

    def calcul(self, couple):

        if isinstance(couple, SerieOfArraysFromFiles):
            couple = ArrayCouple(serie=couple)

        if not isinstance(couple, ArrayCouple):
            raise ValueError

        couple.apply_mask(self.params.mask)

        im0, im1 = couple.get_arrays()

        positions, displacements = optical_flow(
            im0,
            im1,
            self.dict_params_features,
            self.dict_params_flow,
            threshold_diff_ab_ba=self.params.filters.threshold_diff_ab_ba,
            vmin=self.params.filters.displacement_min,
            vmax=self.params.filters.displacement_max,
        )

        xs = positions[:, 0]
        ys = positions[:, 1]
        xs, ys = self._xyoriginalimage_from_xymasked(xs, ys)

        result = HeavyPIVResults(
            deltaxs=displacements[:, 0],
            deltays=displacements[:, 1],
            xs=xs,
            ys=ys,
            couple=couple,
            params=self.params,
        )

        return result
