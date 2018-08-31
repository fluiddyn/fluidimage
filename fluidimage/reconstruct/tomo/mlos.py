"""MLOS (Multiplicative Line of Sight)
======================================

Reference::
    
    C. Atkinson and J. Soria, “An efficient simultaneous reconstruction
    technique for tomographic particle image velocimetry,” Exp Fluids, vol. 47,
    no. 4–5, p. 553, Oct. 2009.


"""

import os
from pathlib import Path
import numpy as np
from scipy import interpolate, sparse
import matplotlib.pyplot as plt
import dask.array as da
from dask.diagnostics import ProgressBar
from psutil import virtual_memory
import cv2

from fluidimage.util.util import imread
from fluidimage.calibration.calib_cv import CalibCV
from fluidimage.data_objects.tomo import ArrayTomoCV


class TomoMLOSBase:
    """MLOS can be summarized in the following steps::
        
        1. Project the world coordinates to pixel coordinates.
        #. Interpolate intensity of the neighbouring pixels.
        #. Project back the interpolated intesities onto world coordinates and
           apply them multiplicatively.

    """

    def __init__(self, cls_calib, cls_array, *cams, **kwargs):
        self.array = cls_array(params=None, **kwargs)
        self.cams = []
        for cam in cams:
            cam_name = Path(cam).name.split(".")[0]
            self.cams.append(cam_name)
            setattr(self, cam_name, cls_calib(cam))

    def reconstruct(self, pix, image, threshold, chunks, save):
        self.array.image_path = image
        interp = self.get_interpolator(image, threshold)
        with ProgressBar():
            x_pix = da.from_array(pix["x"], chunks=chunks)
            y_pix = da.from_array(pix["y"], chunks=chunks)
            i_vox = da.map_blocks(interp, x_pix, y_pix)
            i_vox = i_vox ** (1. / len(self.cams))
            if save:
                if os.path.exists(self.array.h5file_path):
                    # Create a temporary copy to read from
                    f, dset, h5file_tmp = self.array.load_dataset(copy=True)
                    self.array.I = da.from_array(dset, chunks=chunks)
                    self.array.I *= i_vox
                    self.array.save()
                    f.close()
                    os.remove(h5file_tmp)
                else:
                    self.array.I = i_vox
                    self.array.save()
            else:
                self.array.I *= i_vox.compute()

    def verify_projection(self, cam="cam0", skip=1):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        cmap = plt.get_cmap("gray")

        pix = self.phys2pix(cam)

        titles = [r + r"$_{world}$" for r in ("x", "y", "z")]
        for i, axt in enumerate(zip(axes, titles)):
            ax, title = axt
            try:
                r_world = self.array.grid[::skip, i]
            except AttributeError:
                key = ("X", "Y", "Z")[i]
                r_world = getattr(self.array, key).flatten()[::skip]

            color = cmap(r_world / r_world.max())
            ax.scatter(
                pix["x"].flatten()[::skip], pix["y"].flatten()[::skip], c=color
            )
            ax.set_title(title)
            # ax.pcolor(pix["x"], pix["y"], r_world)
            ax.set_xlabel(r"$x_{pix}$")
            ax.set_ylabel(r"$y_{pix}$")


class TomoMLOSRbf(TomoMLOSBase):
    """Interpolation is calculated using a radial basis function (RBF) interpolation
    with a gaussian kernel.

    """

    def get_interpolator(self, image, threshold):
        im = imread(image)

        if threshold is not None:
            im[im < threshold] = 0.
        im_sparse = sparse.coo_matrix(im)

        # Interpolation function
        return interpolate.Rbf(
            im_sparse.col, im_sparse.row, im_sparse.data, function="gaussian"
        )


class TomoMLOSNeighbour(TomoMLOSBase):
    """Interpolation is calculated using a nearest neighbour interpolation.

    """

    def get_interpolator(self, image, threshold):
        im = imread(image)

        if threshold is not None:
            im[im < threshold] = 0.
        im_sparse = sparse.coo_matrix(im)

        # Interpolation function
        return interpolate.NearestNDInterpolator(
            (im_sparse.col, im_sparse.row), im_sparse.data
        )


class TomoMLOSCV(TomoMLOSNeighbour):
    def __init__(self, *cams, **kwargs):
        super().__init__(CalibCV, ArrayTomoCV, *cams, **kwargs)

    def phys2pix(self, cam_name):
        """Tranform the 'physical' world coordinates to 'pixel' coordinates."""
        cam = getattr(self, cam_name)
        grid3d = self.array.grid
        pix = np.empty((len(grid3d), 1, 2))

        for i, z in enumerate(self.array.zs):
            grid_zslice = grid3d[grid3d[:, 2] == z]
            n = len(grid_zslice)
            istart = i * n
            iend = (i + 1) * n
            rotation = cam.get_rotation(z)
            translate = cam.get_translate(z)
            pix[istart:iend, ...], jac = cv2.projectPoints(
                grid_zslice,
                rotation,
                translate,
                cam.params.cam_mtx,
                cam.params.kc,
            )
        return {"x": pix[:, 0, 0], "y": pix[:, 0, 1]}

    def reconstruct(self, pix, image, threshold=None, chunks=None, save=False):
        if chunks is None:
            chunks = len(pix["x"]) // os.cpu_count()
            nmax = int(_estimate_max_array_size(self.array.dtype)) // 6
            if chunks > nmax:
                chunks = nmax

        super().reconstruct(pix, image, threshold, chunks, save)


def _estimate_max_array_size(dtype=np.float64):
    mem = virtual_memory()
    nbytes = np.array([0], dtype=dtype).nbytes
    return mem.available // nbytes


if __name__ == "__main__":
    import shutil
    from fluidimage import path_image_samples

    path = path_image_samples / "TomoPIV" / "calibration"
    cameras = [str(path / f"cam{i}.h5") for i in range(4)]
    tomo = TomoMLOSCV(
        *cameras,
        xlims=(-10, 10),
        ylims=(-10, 10),
        zlims=(-5, 5),
        nb_voxels=(20, 20, 10),
    )
    tomo.verify_projection()

    particle_images = path_image_samples / "TomoPIV" / "particle"
    output_dir = path_image_samples / "TomoPIV" / "tomo"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for cam in tomo.cams:
        print(f"Projecting {cam}...")
        pix = tomo.phys2pix(cam)
        i0 = 1
        for i1 in ["a", "b"]:
            image = str(particle_images / f"{cam}.pre" / f"im{i0:05.0f}{i1}.tif")
            tomo.array.init_paths(image)
            print(f"MLOS of {cam} on {image}: reconstructing...")
            tomo.reconstruct(pix, image, threshold=None, save=True)
