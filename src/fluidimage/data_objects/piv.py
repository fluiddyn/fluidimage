"""PIV objects (:mod:`fluidimage.data_objects.piv`)
===================================================

.. autoclass:: ArrayCouple
   :members:
   :private-members:

.. autoclass:: HeavyPIVResults
   :members:
   :private-members:

.. autoclass:: MultipassPIVResults
   :members:
   :private-members:

.. autoclass:: LightPIVResults
   :members:
   :private-members:

"""

import os

import h5netcdf
import h5py
import numpy as np

from fluidimage import ParamContainer
from fluidimage import __version__ as fluidimage_version
from fluidimage import imread
from fluidimage._version import hg_rev
from fluidimage.util import safe_eval

from .display_piv import DisplayPIV


def get_name_piv(serie, prefix="piv"):
    slicing_tuples = serie.get_slicing_tuples()
    str_indices = ""
    for idim, inds in enumerate(slicing_tuples):
        index = inds[0]
        str_index = serie.get_str_for_name_from_idim_idx(idim, index)
        if len(inds) > 1:
            str_index += "-" + serie.get_str_for_name_from_idim_idx(
                idim, inds[1] - 1
            )

        if idim > 1:
            str_indices += serie.get_index_separators[idim - 1]
        str_indices += str_index

    name = prefix + "_" + str_indices + ".h5"
    return name


def get_name_bos(name, serie):
    name = name[len(serie.base_name) :]
    if serie.extension_file is not None:
        name = name[: -len(serie.extension_file) - 1]
    return "bos" + name + ".h5"


class DataObject:
    pass


def get_slices_from_strcrop(strcrop):
    return tuple(
        slice(*(int(i) if i else None for i in part.strip().split(":")))
        for part in strcrop.split(",")
    )


class ArrayCouple(DataObject):
    """Couple of arrays (images)."""

    def __init__(
        self,
        names=None,
        arrays=None,
        paths=None,
        serie=None,
        str_path=None,
        hdf5_parent=None,
        params_mask=None,
    ):
        self.params_mask = params_mask

        if str_path is not None:
            self._load(path=str_path)
            return

        if hdf5_parent is not None:
            self._load(hdf5_object=hdf5_parent["couple"])
            return

        if serie is not None:
            names = serie.get_name_arrays()
            if len(names) != 2:
                raise ValueError("serie has to contain 2 arrays.")

            paths = serie.get_path_arrays()

            if not serie.check_all_files_exist():
                raise ValueError(
                    "At least one file of this serie does not exists. \n"
                    + str(paths)
                )

            self.paths = paths = tuple(os.path.abspath(p) for p in paths)

            if arrays is None:
                arrays = self.read_images()

        self.paths = paths
        self.names = tuple(names)
        self.name = "-".join(self.names)
        self.arrays = self._mask_arrays(arrays)
        self.serie = serie

    def _read_image(self, index):
        arr = imread(self.paths[index])
        return arr

    def read_images(self):
        return tuple(self._read_image(i) for i in [0, 1])

    def apply_mask(self, params_mask):
        if self.params_mask is not None and params_mask is None:
            raise NotImplementedError

        if self.params_mask is not None and params_mask == self.params_mask:
            return

        if self.params_mask is not None:
            raise NotImplementedError

        self.params_mask = params_mask
        self.arrays = self._mask_arrays(self.arrays)

    def _mask_array(self, array):
        if self.params_mask is None:
            return array

        if self.params_mask.strcrop is not None:
            indices = get_slices_from_strcrop(self.params_mask.strcrop)
            array = array[indices]
        return array

    def _mask_arrays(self, arrays):
        return tuple(self._mask_array(arr) for arr in arrays)

    def get_arrays(self):
        if not hasattr(self, "arrays"):
            self.arrays = self._mask_arrays(self.read_images())

        return self.arrays

    def save(self, path=None, hdf5_parent=None):
        if path is not None:
            raise NotImplementedError

        if not isinstance(hdf5_parent, (h5py.File,)):
            raise NotImplementedError

        hdf5_parent.create_group("couple")
        group = hdf5_parent["couple"]

        assert isinstance(self.names, tuple)

        group.attrs["names"] = repr(self.names).encode()
        group.attrs["paths"] = repr(self.paths).encode()

        if not hasattr(self, "arrays"):
            arr0 = self._mask_array(self._read_image(0))
        else:
            arr0 = self.arrays[0]

        group.create_dataset("shape_images", data=arr0.shape)

    def _load(self, path=None, hdf5_object=None):
        if path is not None:
            raise NotImplementedError

        names = hdf5_object.attrs["names"]
        paths = hdf5_object.attrs["paths"]

        if isinstance(names, bytes):
            names = names.decode()
            paths = paths.decode()

        self.names = safe_eval(names)
        self.paths = safe_eval(paths)

        assert isinstance(self.names, tuple)

        try:
            self.shape_images = hdf5_object["shape_images"][...]
        except KeyError:
            pass


class ArrayCoupleBOS(ArrayCouple):
    """Couple of arrays (images)."""

    def __init__(
        self,
        names=None,
        arrays=None,
        serie=None,
        paths=None,
        str_path=None,
        hdf5_parent=None,
        params_mask=None,
    ):
        self.params_mask = params_mask

        if str_path is not None:
            self._load(path=str_path)
            return

        if hdf5_parent is not None:
            self._load(hdf5_object=hdf5_parent["couple"])
            return

        if paths is not None:
            self.paths = tuple(os.path.abspath(p) for p in paths)

        self.serie = serie
        self.names = tuple(names)
        self.name = self.names[-1]
        self.arrays = self._mask_arrays(arrays)


class HeavyPIVResults(DataObject):
    """Heavy PIV results containing displacements and correlation.

    Attributes
    ----------

    xs, ys: 1d array of size `num_vectors`

      Positions of the vectors in pixels. Depend on the images.

    deltaxs, deltays: 1d array of size `num_vectors`

      Raw PIV results.

    correls: list of `num_vectors` 2d arrays

      Correlation matrices for each vector.

    correls_max: 1d array of size `num_vectors`

      Maximum correlation for each vector.

    deltaxs_approx, deltays_approx: 1d array of size `num_vectors_next_pass`

      Displacements interpolated on a grid that do not depend on the images.

    ixvecs_approx, iyvecs_approx: 1d array of size `num_vectors_next_pass`

      Positions in pixels of the vectors in ``deltaxs_approx``, ``deltays_approx``.

    deltaxs_final, deltays_final, ixvecs_final, iyvecs_final:

      Equivalent to the `_approx` variables but for the last pass
      and of size ``num_vectors``.

    """

    _keys_to_be_saved = [
        "xs",
        "ys",
        "deltaxs",
        "deltays",
        "correls_max",
        "deltaxs_approx",
        "deltays_approx",
        "ixvecs_approx",
        "iyvecs_approx",
        "deltaxs_final",
        "deltays_final",
        "ixvecs_final",
        "iyvecs_final",
    ]

    _dict_to_be_saved = ["errors", "deltaxs_wrong", "deltays_wrong"]

    def __init__(
        self,
        deltaxs=None,
        deltays=None,
        xs=None,
        ys=None,
        errors=None,
        correls_max=None,
        correls=None,
        couple=None,
        params=None,
        str_path=None,
        hdf5_object=None,
        secondary_peaks=None,
        indices_no_displacement=None,
    ):
        if hdf5_object is not None:
            if couple is not None:
                self.couple = couple

            if params is not None:
                self.params = params

            self._load_from_hdf5_object(hdf5_object)
            return

        if str_path is not None:
            self._load(str_path)
            return

        self.deltaxs = deltaxs
        self.deltays = deltays
        self.ys = ys
        self.xs = xs
        self.errors = errors
        self.correls_max = correls_max
        self.correls = correls
        self.couple = couple
        self.params = params
        self.secondary_peaks = secondary_peaks
        self.indices_no_displacement = indices_no_displacement

    def get_images(self):
        return self.couple.read_images()

    def display(
        self,
        show_interp=False,
        scale=0.2,
        show_error=True,
        pourcent_histo=99,
        hist=False,
        show_correl=True,
        xlim=None,
        ylim=None,
    ):
        try:
            im0, im1 = self.get_images()
        except IOError:
            im0, im1 = None, None
        return DisplayPIV(
            im0,
            im1,
            self,
            show_interp=show_interp,
            scale=scale,
            show_error=show_error,
            pourcent_histo=pourcent_histo,
            hist=hist,
            show_correl=show_correl,
            xlim=xlim,
            ylim=ylim,
        )

    def _get_name(self, kind):
        serie = self.couple.serie

        if kind is None:
            if serie is None:
                return self.couple.name + ".h5"
            return get_name_piv(serie, prefix="piv")

        elif kind == "bos":
            return get_name_bos(self.couple.name, serie)

    def save(self, path=None, out_format=None, kind=None):
        name = self._get_name(kind)

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        if out_format == "uvmat":
            with h5netcdf.File(path_file, "w") as file:
                self._save_as_uvmat(file)
        else:
            with h5py.File(path_file, "w") as file:
                file.attrs["class_name"] = "HeavyPIVResults"
                file.attrs["module_name"] = "fluidimage.data_objects.piv"
                self._save_in_hdf5_object(file)

        return path_file

    def _save_in_hdf5_object(self, file, tag="piv0"):
        if "class_name" not in file.attrs.keys():
            file.attrs["class_name"] = "HeavyPIVResults"
            file.attrs["module_name"] = "fluidimage.data_objects.piv"

        if "params" not in file.keys():
            self.params._save_as_hdf5(hdf5_parent=file)

        if "couple" not in file.keys():
            self.couple.save(hdf5_parent=file)

        g_piv = file.create_group(tag)
        g_piv.attrs["class_name"] = "HeavyPIVResults"
        g_piv.attrs["module_name"] = "fluidimage.data_objects.piv"

        for k in self._keys_to_be_saved:
            if k in self.__dict__ and self.__dict__[k] is not None:
                g_piv.create_dataset(k, data=self.__dict__[k])

        for name_dict in self._dict_to_be_saved:
            try:
                d = self.__dict__[name_dict]
                if d is None:
                    raise KeyError
            except KeyError:
                pass
            else:
                g = g_piv.create_group(name_dict)
                keys = list(d.keys())
                values = list(d.values())
                try:
                    for i, k in enumerate(keys):
                        keys[i] = k.encode()
                except AttributeError:
                    pass
                try:
                    for i, k in enumerate(values):
                        values[i] = k.encode()
                except AttributeError:
                    pass

                g.create_dataset("keys", data=keys)
                g.create_dataset("values", data=values)

        if "deltaxs_tps" in self.__dict__:
            g = g_piv.create_group("deltaxs_tps")
            for i, arr in enumerate(self.deltaxs_tps):
                g.create_dataset(f"subdom{i}", data=arr)

            g = g_piv.create_group("deltays_tps")
            for i, arr in enumerate(self.deltays_tps):
                g.create_dataset(f"subdom{i}", data=arr)

    def _load(self, path):
        self.file_name = os.path.basename(path)
        with h5py.File(path, "r") as file:
            self._load_from_hdf5_object(file["piv0"])

    def _load_from_hdf5_object(self, g_piv):
        file = g_piv.parent

        if not hasattr(self, "params"):
            self.params = ParamContainer(hdf5_object=file["params"])
            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None

        if not hasattr(self, "couple"):
            self.couple = ArrayCouple(hdf5_parent=file, params_mask=params_mask)

        for k in self._keys_to_be_saved:
            if k in g_piv:
                dataset = g_piv[k]
                self.__dict__[k] = dataset[:]

        for name_dict in self._dict_to_be_saved:
            try:
                g = g_piv[name_dict]
                keys = g["keys"]
                values = g["values"]
                dictionary = {}
                for key, value in zip(keys, values):
                    try:
                        key = k.decode()
                    except AttributeError:
                        pass
                    try:
                        value = value.decode()
                    except AttributeError:
                        pass
                    dictionary[key] = value
                self.__dict__[name_dict] = dictionary

            except KeyError:
                pass

        if "deltaxs_tps" in g_piv.keys():
            g = g_piv["deltaxs_tps"]
            self.deltaxs_tps = []
            for arr in g.keys():
                self.deltaxs_tps.append(g[arr][...])
            g = g_piv["deltays_tps"]
            self.deltays_tps = []
            for arr in g.keys():
                self.deltays_tps.append(g[arr][...])

    def get_grid_pixel(self, index_pass):
        """Recompute 1d arrays containing the approximate positions of the vectors

        Useful to compute a grid on which we can interpolate the displacement fields.

        Parameters
        ----------

        index_pass: int

          Index of the pass

        Returns
        -------

        xs1d: np.ndarray

          Indices (2nd, direction "x") of the pixel in the image

        ys1d: np.ndarray

          Indices (1st, direction "y") of the pixel in the image

        """
        from ..postproc.piv import get_grid_pixel

        return get_grid_pixel(self.params, self.couple.shape_images, index_pass)


class MultipassPIVResults(DataObject):
    """Result of a multipass PIV computation."""

    def __init__(self, str_path=None):
        self.passes = []

        if str_path is not None:
            self._load(str_path)

    def display(
        self,
        i=-1,
        show_interp=False,
        scale=0.2,
        show_error=True,
        pourcent_histo=99,
        hist=False,
        show_correl=True,
        xlim=None,
        ylim=None,
    ):
        r = self.passes[i]
        return r.display(
            show_interp=show_interp,
            scale=scale,
            show_error=show_error,
            pourcent_histo=pourcent_histo,
            hist=hist,
            show_correl=show_correl,
            xlim=xlim,
            ylim=ylim,
        )

    def __getitem__(self, key):
        return self.passes[key]

    def append(self, results):
        i = len(self.passes)
        self.passes.append(results)
        self.__dict__[f"piv{i}"] = results

    def _get_name(self, kind):
        if hasattr(self, "file_name"):
            return self.file_name

        r = self.passes[0]
        return r._get_name(kind)

    def save(self, path=None, out_format=None, kind=None):
        name = self._get_name(kind)

        if path is not None:
            root, ext = os.path.splitext(path)
            if ext in [".h5", ".nc"]:
                path_file = path
            else:
                path_file = os.path.join(path, name)
        else:
            path_file = name

        if out_format == "uvmat":
            with h5netcdf.File(path_file, "w") as file:
                self._save_as_uvmat(file)
        else:
            with h5py.File(path_file, "w") as file:
                file.attrs["class_name"] = "MultipassPIVResults"
                file.attrs["module_name"] = "fluidimage.data_objects.piv"

                file.attrs["nb_passes"] = len(self.passes)

                file.attrs["fluidimage_version"] = fluidimage_version
                file.attrs["fluidimage_hg_rev"] = hg_rev

                for i, r in enumerate(self.passes):
                    r._save_in_hdf5_object(file, tag=f"piv{i}")

        return path_file

    def _load(self, path):
        self.file_name = os.path.basename(path)
        with h5py.File(path, "r") as file:
            self.params = ParamContainer(hdf5_object=file["params"])

            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None

            self.couple = ArrayCouple(hdf5_parent=file, params_mask=params_mask)

            nb_passes = file.attrs["nb_passes"]

            for ip in range(nb_passes):
                g = file[f"piv{ip}"]
                self.append(
                    HeavyPIVResults(
                        hdf5_object=g, couple=self.couple, params=self.params
                    )
                )

    def _save_as_uvmat(self, file):
        file.dimensions = {"nb_coord": 2, "nb_bounds": 2}

        for i, ir in enumerate(self.passes):
            iuvmat = i + 1
            str_i = str(iuvmat)
            str_nb_vec = "nb_vec_" + str_i
            file.dimensions[str_nb_vec] = ir.xs.size

            tmp = np.zeros(ir.deltaxs.shape).astype("float32")
            inds = np.where(~np.isnan(ir.deltaxs))

            file.create_variable(f"Civ{iuvmat}_X", (str_nb_vec,), data=ir.xs)
            file.create_variable(f"Civ{iuvmat}_Y", (str_nb_vec,), data=ir.ys)
            file.create_variable(
                f"Civ{iuvmat}_U", (str_nb_vec,), data=np.nan_to_num(ir.deltaxs)
            )
            file.create_variable(
                f"Civ{iuvmat}_V", (str_nb_vec,), data=np.nan_to_num(ir.deltays)
            )

            if ir.params.multipass.use_tps:
                str_nb_subdom = f"nb_subdomain_{iuvmat}"
                try:
                    file.dimensions[str_nb_subdom] = np.shape(ir.deltaxs_tps)[0]
                    file.dimensions[f"nb_tps{iuvmat}"] = np.shape(ir.deltaxs_tps)[
                        1
                    ]
                    tmp[inds] = ir.deltaxs_smooth
                    file.create_variable(
                        f"Civ{iuvmat}_U_smooth", (str_nb_vec,), data=tmp
                    )
                    tmp[inds] = ir.deltays_smooth
                    file.create_variable(
                        f"Civ{iuvmat}_V_smooth", (str_nb_vec,), data=tmp
                    )
                    file.create_variable(
                        f"Civ{iuvmat}_U_tps",
                        (str_nb_subdom, f"nb_vec_tps_{iuvmat}"),
                        data=ir.deltaxs_tps,
                    )
                    file.create_variable(
                        f"Civ{iuvmat}_V_tps",
                        (str_nb_subdom, f"nb_vec_tps_{iuvmat}"),
                        data=ir.deltays_tps,
                    )
                    tmp = [None] * file.dimensions[str_nb_subdom]
                    for j in range(file.dimensions[str_nb_subdom]):
                        tmp[j] = np.shape(ir.deltaxs_tps[j])[0]
                    file.create_variable(
                        f"Civ{iuvmat}_NbCentres",
                        (f"nb_subdomain_{iuvmat}",),
                        data=tmp,
                    )
                except:
                    print(f"no tps field at passe n {iuvmat}")

            file.create_variable(
                f"Civ{iuvmat}_C", (str_nb_vec,), data=ir.correls_max
            )
            tmp = np.zeros(ir.deltaxs.shape).astype("float32")
            indsnan = np.where(np.isnan(ir.deltaxs))
            tmp[indsnan] = 1
            file.create_variable(f"Civ{iuvmat}_FF", (str_nb_vec,), data=tmp)

            # mettre bonne valeur de F correspondant a self.piv0.error...
            file.create_variable(f"Civ{iuvmat}_F", (str_nb_vec,), data=tmp)

    # ADD
    # file.create_variable('Civ1_Coord_tps',
    #                   ('nb_subdomain_1', 'nb_coord', 'nb_tps_1'),
    #                   data=???)

    # ADD attributes

    def make_light_result(self, ind_pass=-1):
        piv = self.passes[ind_pass]
        if ind_pass == -1:
            deltaxs = piv.deltaxs_final
            deltays = piv.deltays_final
            ixvec = piv.ixvecs_final
            iyvec = piv.iyvecs_final

        else:
            deltaxs = piv.deltaxs_approx
            deltays = piv.deltays_approx
            ixvec = piv.ixvecs_approx
            iyvec = piv.ixvecs_approx

        if hasattr(self, "file_name"):
            file_name = self.file_name
        else:
            file_name = None
        return LightPIVResults(
            deltaxs,
            deltays,
            ixvec,
            iyvec,
            couple=piv.couple,
            params=piv.params,
            file_name=file_name,
        )

    def get_grid_pixel(self, index_pass=-1):
        """Recompute 1d arrays containing the approximate positions of the vectors

        Useful to compute a grid on which we can interpolate the displacement fields.

        Parameters
        ----------

        index_pass: int

          Index of the pass

        Returns
        -------

        xs1d: np.ndarray

          Indices (2nd, direction "x") of the pixel in the image

        ys1d: np.ndarray

          Indices (1st, direction "y") of the pixel in the image

        """
        piv = self.passes[index_pass]
        return piv.get_grid_pixel(index_pass)


class LightPIVResults(DataObject):
    _keys_to_be_saved = [
        "ixvecs_final",
        "iyvecs_final",
        "deltaxs_final",
        "deltays_final",
    ]

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __init__(
        self,
        deltaxs_approx=None,
        deltays_approx=None,
        ixvecs_grid=None,
        iyvecs_grid=None,
        correls_max=None,
        couple=None,
        params=None,
        str_path=None,
        hdf5_object=None,
        file_name=None,
    ):
        if file_name is not None:
            self.file_name = file_name
        if hdf5_object is not None:
            if couple is not None:
                self.couple = couple

            if params is not None:
                self.params = params

            self._load_from_hdf5_object(hdf5_object)
            return

        if str_path is not None:
            self._load(str_path)
            return

        self.deltaxs_final = deltaxs_approx
        self.deltays_final = deltays_approx
        self.couple = couple
        self.params = params
        self.ixvecs_final = ixvecs_grid
        self.iyvecs_final = iyvecs_grid

    def _get_name(self, kind):
        if hasattr(self, "file_name"):
            return self.file_name[:-3] + "_light.h5"

        serie = self.couple.serie

        str_ind0 = serie.compute_str_indices_from_indices(
            *[inds[0] for inds in serie.get_slicing_tuples()]
        )

        str_ind1 = serie.compute_str_indices_from_indices(
            *[inds[1] - 1 for inds in serie.get_slicing_tuples()]
        )

        name = "piv_" + serie.base_name + str_ind0 + "-" + str_ind1 + "_light.h5"
        return name

    def save(self, path=None, out_format=None, kind=None):
        name = self._get_name(kind)

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        with h5py.File(path_file, "w") as file:
            file.attrs["class_name"] = "LightPIVResults"
            file.attrs["module_name"] = "fluidimage.data_objects.piv"

            self._save_in_hdf5_object(file, tag="piv")

        return self

    def _save_in_hdf5_object(self, file, tag="piv"):
        if "class_name" not in file.attrs.keys():
            file.attrs["class_name"] = "LightPIVResults"
            file.attrs["module_name"] = "fluidimage.data_objects.piv"
        if "params" not in file.keys():
            self.params._save_as_hdf5(hdf5_parent=file)
        if "couple" not in file.keys():
            self.couple.save(hdf5_parent=file)

        g_piv = file.create_group(tag)
        g_piv.attrs["class_name"] = "LightPIVResults"
        g_piv.attrs["module_name"] = "fluidimage.data_objects.piv"

        for k in self._keys_to_be_saved:
            if k in self.__dict__:
                g_piv.create_dataset(k, data=self.__dict__[k])

    def _load(self, path):
        with h5py.File(path, "r") as file:
            self.params = ParamContainer(hdf5_object=file["params"])
            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None
            self.couple = ArrayCouple(hdf5_parent=file, params_mask=params_mask)

        with h5py.File(path, "r") as file:
            keys = [(key) for key in file.keys() if "piv" in key]
            self._load_from_hdf5_object(file[max(keys)])

    def _load_from_hdf5_object(self, g_piv):
        file = g_piv.parent

        if not hasattr(self, "params"):
            self.params = ParamContainer(hdf5_object=file["params"])
            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None

        if not hasattr(self, "couple"):
            self.couple = ArrayCouple(hdf5_parent=file, params_mask=params_mask)

        for k in self._keys_to_be_saved:
            if k in g_piv:
                dataset = g_piv[k]
                self.__dict__[k] = dataset[:]
