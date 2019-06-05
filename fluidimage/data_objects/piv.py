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

from .display_piv import DisplayPIV
from .. import ParamContainer
from .. import __version__ as fluidimage_version
from .. import imread
from .._hg_rev import hg_rev


def get_str_index(serie, i, index):
    if serie._from_movies and i == serie.nb_indices - 1:
        return str(index)

    if serie._index_types[i] == "digit":
        code_format = "{:0" + str(serie._index_lens[i]) + "d}"
        str_index = code_format.format(index)
    elif serie._index_types[i] == "alpha":
        if index > 25:
            raise ValueError('"alpha" index larger than 25.')

        str_index = chr(ord("a") + index)
    else:
        raise Exception('The type should be "digit" or "alpha".')

    return str_index


def get_name_piv(serie, prefix="piv"):
    index_slices = serie._index_slices
    str_indices = ""
    for i, inds in enumerate(index_slices):
        index = inds[0]
        str_index = get_str_index(serie, i, index)
        if len(inds) > 1:
            str_index += "-" + get_str_index(serie, i, inds[1] - 1)

        if i > 1:
            str_indices += serie._index_separators[i - 1]
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

            self.paths = tuple(os.path.abspath(p) for p in paths)

            if arrays is None:
                arrays = self.read_images()

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

        names = hdf5_object.attrs["names"].decode()
        paths = hdf5_object.attrs["paths"].decode()

        self.names = eval(names)
        self.paths = eval(paths)

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
    """Heavy PIV results containing displacements and correlation."""

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
        )

    def _get_name(self, kind):
        serie = self.couple.serie

        if kind is None:
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
            with h5netcdf.File(path_file, "w") as f:
                self._save_as_uvmat(f)
        else:
            with h5py.File(path_file, "w") as f:
                f.attrs["class_name"] = "HeavyPIVResults"
                f.attrs["module_name"] = "fluidimage.data_objects.piv"
                self._save_in_hdf5_object(f)

        return path_file

    def _save_in_hdf5_object(self, f, tag="piv0"):

        if "class_name" not in f.attrs.keys():
            f.attrs["class_name"] = "HeavyPIVResults"
            f.attrs["module_name"] = "fluidimage.data_objects.piv"

        if "params" not in f.keys():
            self.params._save_as_hdf5(hdf5_parent=f)

        if "couple" not in f.keys():
            self.couple.save(hdf5_parent=f)

        g_piv = f.create_group(tag)
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
        with h5py.File(path, "r") as f:
            self._load_from_hdf5_object(f["piv0"])

    def _load_from_hdf5_object(self, g_piv):

        f = g_piv.parent

        if not hasattr(self, "params"):
            self.params = ParamContainer(hdf5_object=f["params"])
            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None

        if not hasattr(self, "couple"):
            self.couple = ArrayCouple(hdf5_parent=f, params_mask=params_mask)

        for k in self._keys_to_be_saved:
            if k in g_piv:
                dataset = g_piv[k]
                self.__dict__[k] = dataset[:]

        for name_dict in self._dict_to_be_saved:
            try:
                g = g_piv[name_dict]
                keys = g["keys"]
                values = g["values"]
                self.__dict__[name_dict] = {k: v for k, v in zip(keys, values)}
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
    ):
        r = self.passes[i]
        return r.display(
            show_interp=show_interp,
            scale=scale,
            show_error=show_error,
            pourcent_histo=pourcent_histo,
            hist=hist,
            show_correl=show_correl,
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
            with h5netcdf.File(path_file, "w") as f:
                self._save_as_uvmat(f)
        else:
            with h5py.File(path_file, "w") as f:
                f.attrs["class_name"] = "MultipassPIVResults"
                f.attrs["module_name"] = "fluidimage.data_objects.piv"

                f.attrs["nb_passes"] = len(self.passes)

                f.attrs["fluidimage_version"] = fluidimage_version
                f.attrs["fluidimage_hg_rev"] = hg_rev

                for i, r in enumerate(self.passes):
                    r._save_in_hdf5_object(f, tag=f"piv{i}")

        return path_file

    def _load(self, path):

        self.file_name = os.path.basename(path)
        with h5py.File(path, "r") as f:
            self.params = ParamContainer(hdf5_object=f["params"])

            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None

            self.couple = ArrayCouple(hdf5_parent=f, params_mask=params_mask)

            nb_passes = f.attrs["nb_passes"]

            for ip in range(nb_passes):
                g = f[f"piv{ip}"]
                self.append(
                    HeavyPIVResults(
                        hdf5_object=g, couple=self.couple, params=self.params
                    )
                )

    def _save_as_uvmat(self, f):

        f.dimensions = {"nb_coord": 2, "nb_bounds": 2}

        for i, ir in enumerate(self.passes):

            iuvmat = i + 1
            str_i = str(iuvmat)
            str_nb_vec = "nb_vec_" + str_i
            f.dimensions[str_nb_vec] = ir.xs.size

            tmp = np.zeros(ir.deltaxs.shape).astype("float32")
            inds = np.where(~np.isnan(ir.deltaxs))

            f.create_variable(f"Civ{iuvmat}_X", (str_nb_vec,), data=ir.xs)
            f.create_variable(f"Civ{iuvmat}_Y", (str_nb_vec,), data=ir.ys)
            f.create_variable(
                f"Civ{iuvmat}_U", (str_nb_vec,), data=np.nan_to_num(ir.deltaxs)
            )
            f.create_variable(
                f"Civ{iuvmat}_V", (str_nb_vec,), data=np.nan_to_num(ir.deltays)
            )

            if ir.params.multipass.use_tps:
                str_nb_subdom = f"nb_subdomain_{iuvmat}"
                try:
                    f.dimensions[str_nb_subdom] = np.shape(ir.deltaxs_tps)[0]
                    f.dimensions[f"nb_tps{iuvmat}"] = np.shape(ir.deltaxs_tps)[1]
                    tmp[inds] = ir.deltaxs_smooth
                    f.create_variable(
                        f"Civ{iuvmat}_U_smooth", (str_nb_vec,), data=tmp
                    )
                    tmp[inds] = ir.deltays_smooth
                    f.create_variable(
                        f"Civ{iuvmat}_V_smooth", (str_nb_vec,), data=tmp
                    )
                    f.create_variable(
                        f"Civ{iuvmat}_U_tps",
                        (str_nb_subdom, f"nb_vec_tps_{iuvmat}"),
                        data=ir.deltaxs_tps,
                    )
                    f.create_variable(
                        f"Civ{iuvmat}_V_tps",
                        (str_nb_subdom, f"nb_vec_tps_{iuvmat}"),
                        data=ir.deltays_tps,
                    )
                    tmp = [None] * f.dimensions[str_nb_subdom]
                    for j in range(f.dimensions[str_nb_subdom]):
                        tmp[j] = np.shape(ir.deltaxs_tps[j])[0]
                    f.create_variable(
                        f"Civ{iuvmat}_NbCentres",
                        (f"nb_subdomain_{iuvmat}",),
                        data=tmp,
                    )
                except:
                    print(f"no tps field at passe n {iuvmat}")

            f.create_variable(
                f"Civ{iuvmat}_C", (str_nb_vec,), data=ir.correls_max
            )
            tmp = np.zeros(ir.deltaxs.shape).astype("float32")
            indsnan = np.where(np.isnan(ir.deltaxs))
            tmp[indsnan] = 1
            f.create_variable(f"Civ{iuvmat}_FF", (str_nb_vec,), data=tmp)

            # mettre bonne valeur de F correspondant a self.piv0.error...
            f.create_variable(f"Civ{iuvmat}_F", (str_nb_vec,), data=tmp)

    # ADD
    # f.create_variable('Civ1_Coord_tps',
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

        str_ind0 = serie._compute_strindices_from_indices(
            *[inds[0] for inds in serie.get_index_slices()]
        )

        str_ind1 = serie._compute_strindices_from_indices(
            *[inds[1] - 1 for inds in serie.get_index_slices()]
        )

        name = "piv_" + serie.base_name + str_ind0 + "-" + str_ind1 + "_light.h5"
        return name

    def save(self, path=None, out_format=None, kind=None):
        name = self._get_name(kind)

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        with h5py.File(path_file, "w") as f:
            f.attrs["class_name"] = "LightPIVResults"
            f.attrs["module_name"] = "fluidimage.data_objects.piv"

            self._save_in_hdf5_object(f, tag="piv")

        return self

    def _save_in_hdf5_object(self, f, tag="piv"):

        if "class_name" not in f.attrs.keys():
            f.attrs["class_name"] = "LightPIVResults"
            f.attrs["module_name"] = "fluidimage.data_objects.piv"
        if "params" not in f.keys():
            self.params._save_as_hdf5(hdf5_parent=f)
        if "couple" not in f.keys():
            self.couple.save(hdf5_parent=f)

        g_piv = f.create_group(tag)
        g_piv.attrs["class_name"] = "LightPIVResults"
        g_piv.attrs["module_name"] = "fluidimage.data_objects.piv"

        for k in self._keys_to_be_saved:
            if k in self.__dict__:
                g_piv.create_dataset(k, data=self.__dict__[k])

    def _load(self, path):
        with h5py.File(path, "r") as f:
            self.params = ParamContainer(hdf5_object=f["params"])
            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None
            self.couple = ArrayCouple(hdf5_parent=f, params_mask=params_mask)

        with h5py.File(path, "r") as f:
            keys = [(key) for key in f.keys() if "piv" in key]
            self._load_from_hdf5_object(f[max(keys)])

    def _load_from_hdf5_object(self, g_piv):

        f = g_piv.parent

        if not hasattr(self, "params"):
            self.params = ParamContainer(hdf5_object=f["params"])
            try:
                params_mask = self.params.mask
            except AttributeError:
                params_mask = None

        if not hasattr(self, "couple"):
            self.couple = ArrayCouple(hdf5_parent=f, params_mask=params_mask)

        for k in self._keys_to_be_saved:
            if k in g_piv:
                dataset = g_piv[k]
                self.__dict__[k] = dataset[:]
