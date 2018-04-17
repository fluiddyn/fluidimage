
from __future__ import print_function

import unittest

import numpy as np

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import correlation_classes
import pylab

classes = {k.replace(".", "_"): v for k, v in correlation_classes.items()}

try:
    from reikna.cluda import any_api

    api = any_api()
except Exception:
    classes.pop("cufft")

try:
    import pycuda
except ImportError:
    classes.pop("pycuda")

try:
    import skcuda
except ImportError:
    classes.pop("skcufft")

try:
    import theano
except ImportError:
    classes.pop("theano")


def setUpImages(nx, ny, dx, dy, part_size):

    displacements = np.array([dx, dy])

    nb_particles = (nx // 4) ** 2

    im0, im1 = make_synthetic_images(
        displacements,
        nb_particles,
        shape_im0=(ny, nx),
        epsilon=0.,
        part_size=part_size,
    )

    return displacements, im0, im1


def compute_displacement(cls, im0, im1, nsubpix, method_subpix):
    correl = cls(
        im0.shape, im1.shape, method_subpix=method_subpix, nsubpix=nsubpix
    )

    c, norm = correl(im0, im1)
    dx, dy, correl_max = correl.compute_displacement_from_correl(
        c, coef_norm=norm
    )
    displacement_computed = np.array([dx, dy])

    return displacement_computed


def error_vs_nsubpix(nx, ny, dx, dy, method_subpix, n_subpix, part_size):
    displacements, im0, im1 = setUpImages(nx, ny, dx, dy, part_size)
    # pylab.imshow(im0)
    # pylab.show()

    errdx = np.zeros([np.shape(classes.items())[0], n_subpix.size])
    errdy = np.zeros([np.shape(classes.items())[0], n_subpix.size])

    temp = np.reshape(classes.items(), [6, 2])
    leg = temp.T[0]

    indn = 0
    for nsubpix in n_subpix:
        indk = 0
        for k, cls in classes.items():
            try:
                displacement_computed = compute_displacement(
                    cls, im0, im1, nsubpix, method_subpix
                )
            except:
                displacement_computed = np.array([np.nan, np.nan])
            errdx[indk][indn] = np.abs(displacement_computed - displacements)[0]
            errdy[indk][indn] = np.abs(displacement_computed - displacements)[1]

            indk += 1
        indn += 1

    pylab.figure
    for i, legi in enumerate(leg):
        pylab.plot(n_subpix, np.sqrt(errdx[i] ** 2 + errdy[i] ** 2), "o")
    pylab.legend(leg)
    pylab.xlabel("nsubpix")
    pylab.ylabel("error in pix")
    title = (
        method_subpix
        + " _part_size={}".format(part_size)
        + " _nx_ny = {}_{}".format(nx, ny)
    )
    pylab.title(title)
    pylab.xlim([0, np.max(n_subpix) + 1])
    pylab.show()
    pylab.savefig("./plot_evaluate_subpix/" + title + ".png")

    return errdx, errdy


if __name__ == "__main__":
    part_size = 4.0
    nx = ny = 64
    dx = 3.3
    dy = 3.7
    # method_subpix = '2d_gaussian'
    method_subpix = "centroid"
    n_subpix = np.arange(1, 8)
    errdx, errdy = error_vs_nsubpix(
        nx, ny, dx, dy, method_subpix, n_subpix, part_size
    )
