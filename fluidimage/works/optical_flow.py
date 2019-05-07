import numpy as np
import cv2

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

from ..data_objects.piv import ArrayCouple
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

    norm = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
    cond = vmin < norm < vmax
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

WinSize characterize the size of the window in which we search movement.
The algorithm is pyramidal."""
        )

        params._set_child(
            "features",
            attribs=dict(
                maxCorners=70000, qualityLevel=0.09, minDistance=4, blockSize=16
            ),
        )
        params.features._set_doc(
            """Parameters for the Good Feature to Track algorithm (Shi-Tomasi Corner Detector)
the more we consider corners, the more we are able to reproduce the velocity
be careful that with a too low quality level for vectors the results are poor.
Filters are needed to exclude bad vectors."""
        )

        cls._complete_params_with_default_mask(params)

        params._set_child(
            "filter",
            attribs={
                "threshold_diff_ab_ba": 1.0,
                "displacement_min": None,
                "displacement_max": None,
            },
        )

        params.fix._set_doc(
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
        xs, xy = self._xyoriginalimage_from_xymasked(xs, ys)

        return xs, xy, displacements
