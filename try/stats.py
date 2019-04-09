"""
Compute image statistics about a set of center coordinates.

Usage: python -m fluidimage.util.stats image1 image2 [x1 y1 x2 y2 ..]

"""
import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt

try:
    from ..preproc._toolbox_cv import adaptive_threshold
except ImportError:
    from ..preproc._toolbox_py import adaptive_threshold


def particle_count(img):
    positions, count = nd.measurements.label(img)
    return count


def particle_density(img, center_indices=[], window_size=11, di_range=(10, 100)):
    """Plots particle density in the image array about a centre versus
    a range of possible interrogation window sizes.

    Parameters
    ----------
    img : array-like
        2-D image
    center_indices : list of tuples
        A list of centers about which particle density is to be calculated
    window_size : scalar
        Sets the size of the pixel neighbourhood to calculate threshold.
    di_range: tuple
        Tuple indicating minima and maxima of interrogation window sizes.

    """
    img = adaptive_threshold(img, window_size)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Particle density ($N_I$) vs interrogation window size ($D_I$)")
    ax.set_xlabel("$D_I$")
    ax.set_ylabel("$N_I$")

    di_range = np.arange(*di_range, step=2)
    if len(center_indices) == 0:
        center_indices = [np.array(img.shape) // 2]

    for i0, i1 in center_indices:
        ni = np.zeros_like(di_range)
        for i, di in enumerate(di_range):
            di2 = di // 2
            slice0 = slice(i0 - di2, i0 + di2)
            slice1 = slice(i1 - di2, i1 + di2)
            ni[i] = particle_count(img[slice0, slice1])

        ax.plot(di_range, ni, label=f"centre: {i0},{i1}")
        ax.legend()

    plt.show()


def particle_motion_factor(
    img1, img2, center_indices=[], window_size=11, di_range=(10, 100)
):
    """Plots particle factor due to in- and out-of-plane motion versus
    a range of possible interrogation window sizes.

    Parameters
    ----------
    img1, img2 : array-like
        2-D images
    center_indices : list of tuples
        A list of centers about which particle density is to be calculated
    window_size : scalar
        Sets the size of the pixel neighbourhood to calculate threshold.
    di_range: tuple
        Tuple indicating minima and maxima of interrogation window sizes.

    """
    img1 = adaptive_threshold(img1, window_size)
    img2 = adaptive_threshold(img2, window_size)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title(
        "Particle motion factor ($F_{IO}$)" "vs interrogation window size ($D_I$)"
    )
    ax.set_xlabel("$D_I$")
    ax.set_ylabel("$F_{I0}$")

    di_range = np.arange(*di_range, step=2)
    if len(center_indices) == 0:
        center_indices = [np.array(img1.shape) // 2]

    for i0, i1 in center_indices:
        ni1 = np.zeros_like(di_range)
        ni2 = np.zeros_like(di_range)
        for i, di in enumerate(di_range):
            di2 = di // 2
            slice0 = slice(i0 - di2, i0 + di2)
            slice1 = slice(i1 - di2, i1 + di2)
            ni1[i] = particle_count(img1[slice0, slice1])
            ni2[i] = particle_count(img2[slice0, slice1])

        fio = ni2.astype(float) / ni1
        ax.plot(di_range, fio, label=f"centre: {i0},{i1}")
        ax.legend()

    plt.show()


if __name__ == "__main__":
    import sys
    from fluiddyn.io.image import imread

    img1 = imread(sys.argv[1])
    img2 = imread(sys.argv[2])

    argv = map(int, sys.argv[3:])
    center_indices = list(zip(argv[0::2], argv[1::2]))
    print(sys.argv)
    print(center_indices)
    particle_density(img1, center_indices)
    particle_motion_factor(img1, img2, center_indices)
