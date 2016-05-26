"""PIV objects
==============

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

import h5py
import h5netcdf

import numpy as np

from .display import display
from .. import imread, ParamContainer


def get_name_piv(serie, prefix='piv'):
    str_ind0 = serie._compute_strindices_from_indices(
        *[inds[0] for inds in serie.get_index_slices()])

    str_ind1 = serie._compute_strindices_from_indices(
        *[inds[1] - 1 for inds in serie.get_index_slices()])

    name = (prefix + '_' + serie.base_name + str_ind0 + '-' + str_ind1 + '.h5')
    return name


class DataObject(object):
    pass


class ArrayCouple(DataObject):
    def __init__(
            self, names=None, arrays=None, serie=None,
            str_path=None, hdf5_parent=None):

        if str_path is not None:
            self._load(path=str_path)
            return

        if hdf5_parent is not None:
            self._load(hdf5_object=hdf5_parent['couple'])
            return

        if serie is not None:
            if serie.get_nb_files() != 2:
                raise ValueError('serie has to contain 2 arrays.')
            names = serie.get_name_files()
            paths = serie.get_path_files()
            self.paths = tuple(os.path.abspath(p) for p in paths)

            if arrays is None:
                arrays = serie.get_arrays()

        self.names = tuple(names)
        self.arrays = tuple(arrays)
        self.serie = serie

    def get_arrays(self):
        if not hasattr(self, 'arrays'):
            self.arrays = tuple(imread(path) for path in self.paths)

        return self.arrays

    def save(self, path=None, hdf5_parent=None):
        if path is not None:
            raise NotImplementedError

        if not isinstance(hdf5_parent, (h5py.File,)):
            raise NotImplementedError

        hdf5_parent.create_group('couple')
        group = hdf5_parent['couple']
        group.attrs['names'] = self.names
        group.attrs['paths'] = self.paths

    def _load(self, path=None, hdf5_object=None):

        if path is not None:
            raise NotImplementedError

        self.names = tuple(hdf5_object.attrs['names'])
        self.paths = tuple(hdf5_object.attrs['paths'])


class HeavyPIVResults(DataObject):

    def __init__(self, deltaxs=None, deltays=None,
                 xs=None, ys=None, errors=None,
                 correls_max=None, correls=None,
                 couple=None, params=None,
                 str_path=None, hdf5_object=None):
        self._keys_to_be_saved = [
            'xs', 'ys', 'deltaxs', 'deltays', 'correls_max', 'deltaxs_approx',
            'deltays_approx', 'new_positions', 'ixvecs_grid', 'iyvecs_grid']
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

    def get_images(self):
        return self.couple.get_arrays()

    def display(self):
        im0, im1 = self.couple.get_arrays()
        return display(
            im0, im1, self)

    def _get_name(self):
        serie = self.couple.serie
        return get_name_piv(serie, prefix='piv')

    def save(self, path=None, out_format='uvmat'):

        name = self._get_name()

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        if out_format == 'uvmat':
            with h5netcdf.File(path_file, 'w') as f:
                self._save_as_uvmat(f)
        else:
            with h5py.File(path_file, 'w') as f:
                f.attrs['class_name'] = 'MultipassPIVResults'
                f.attrs['module_name'] = 'fluidimage.data_objects.piv'

                for i, r in enumerate(self.passes):
                    r._save_in_hdf5_object(f, tag='piv{}'.format(i))

        return self

    def _save_in_hdf5_object(self, f, tag='piv0'):

        if 'class_name' not in f.attrs.keys():
            f.attrs['class_name'] = 'HeavyPIVResults'
            f.attrs['module_name'] = 'fluidimage.data_objects.piv'

        if 'params' not in f.keys():
            self.params._save_as_hdf5(hdf5_parent=f)

        if 'couple' not in f.keys():
            self.couple.save(hdf5_parent=f)

        g_piv = f.create_group(tag)
        g_piv.attrs['class_name'] = 'HeavyPIVResults'
        g_piv.attrs['module_name'] = 'fluidimage.data_objects.piv'

        for k in self._keys_to_be_saved:
            if k in self.__dict__:
                g_piv.create_dataset(k, data=self.__dict__[k])

        g = g_piv.create_group('errors')
        g.create_dataset('keys', data=self.errors.keys())
        g.create_dataset('values', data=self.errors.values())

        if 'deltaxs_tps' in self.__dict__:
            g = g_piv.create_group('deltaxs_tps')
            for i, arr in enumerate(self.deltaxs_tps):
                g.create_dataset('subdom{}'.format(i), data=arr)

            g = g_piv.create_group('deltays_tps')
            for i, arr in enumerate(self.deltays_tps):
                g.create_dataset('subdom{}'.format(i), data=arr)
                
        
    def _load(self, path):

        self.file_name = os.path.basename(path)
        with h5py.File(path, 'r') as f:
            self._load_from_hdf5_object(f['piv0'])

    def _load_from_hdf5_object(self, g_piv):

        f = g_piv.parent

        if not hasattr(self, 'params'):
            self.params = ParamContainer(hdf5_object=f['params'])

        if not hasattr(self, 'couple'):
            self.couple = ArrayCouple(hdf5_parent=f)

        for k in self._keys_to_be_saved:
            if k in g_piv:
                dataset = g_piv[k]
                self.__dict__[k] = dataset[:]

        g = g_piv['errors']
        keys = g['keys']
        values = g['values']
        self.errors = {k: v for k, v in zip(keys, values)}

        if 'deltaxs_tps' in g_piv.keys():                       
            g = g_piv['deltaxs_tps']
            self.deltaxs_tps = []
            for arr in g.keys():
                self.deltaxs_tps.append(g[arr].value)
            g = g_piv['deltays_tps']
            self.deltays_tps = []
            for arr in g.keys():
                self.deltays_tps.append(g[arr].value)
                
class MultipassPIVResults(DataObject):

    def __init__(self, str_path=None):
        self.passes = []

        if str_path is not None:
            self._load(str_path)

    def display(self, i=-1):
        r = self.passes[i]
        return r.display()

    def __getitem__(self, key):
        return self.passes[key]

    def append(self, results):
        i = len(self.passes)
        self.passes.append(results)
        self.__dict__['piv{}'.format(i)] = results

    def _get_name(self):

        if hasattr(self, 'file_name'):
            return self.file_name

        r = self.passes[0]
        return r._get_name()

    def save(self, path=None, out_format=None):

        name = self._get_name()

        if path is not None:
            root, ext = os.path.splitext(path)
            if ext in ['.h5', '.nc']:
                path_file = path
            else:
                path_file = os.path.join(path, name)
        else:
            path_file = name

        if out_format == 'uvmat':
            with h5netcdf.File(path_file, 'w') as f:
                self._save_as_uvmat(f)
        else:
            with h5py.File(path_file, 'w') as f:
                f.attrs['class_name'] = 'MultipassPIVResults'
                f.attrs['module_name'] = 'fluidimage.data_objects.piv'

                f.attrs['nb_passes'] = len(self.passes)

                for i, r in enumerate(self.passes):
                    r._save_in_hdf5_object(f, tag='piv{}'.format(i))

    def _load(self, path):

        self.file_name = os.path.basename(path)
        with h5py.File(path, 'r') as f:
            self.params = ParamContainer(hdf5_object=f['params'])
            self.couple = ArrayCouple(hdf5_parent=f)

            nb_passes = f.attrs['nb_passes']

            for ip in range(nb_passes):
                g = f['piv{}'.format(ip)]
                self.append(HeavyPIVResults(
                    hdf5_object=g, couple=self.couple, params=self.params))

    def _save_as_uvmat(self, f):

        f.dimensions = {'nb_coord': 2,
                        'nb_bounds': 2}

        for i, ir in enumerate(self.passes):

            iuvmat = i + 1
            str_i = str(iuvmat)
            str_nb_vec = 'nb_vec_' + str_i
            f.dimensions[str_nb_vec] = ir.xs.size

            tmp = np.zeros(ir.deltaxs.shape).astype('float32')
            inds = np.where(~np.isnan(ir.deltaxs))

            f.create_variable(
                'Civ{}_X'.format(iuvmat), (str_nb_vec,),
                data=ir.xs)
            f.create_variable(
                'Civ{}_Y'.format(iuvmat), (str_nb_vec,),
                data=ir.ys)
            f.create_variable(
                'Civ{}_U'.format(iuvmat), (str_nb_vec,),
                data=np.nan_to_num(ir.deltaxs))
            f.create_variable(
                'Civ{}_V'.format(iuvmat), (str_nb_vec,),
                data=np.nan_to_num(ir.deltays))

            if ir.params.multipass.use_tps:
                str_nb_subdom = 'nb_subdomain_{}'.format(iuvmat)
                try:
                    f.dimensions[str_nb_subdom] = \
                        np.shape(ir.deltaxs_tps)[0]
                    f.dimensions['nb_tps{}'.format(iuvmat)] = \
                        np.shape(ir.deltaxs_tps)[1]
                    tmp[inds] = ir.deltaxs_smooth
                    f.create_variable('Civ{}_U_smooth'.format(iuvmat),
                                      (str_nb_vec,), data=tmp)
                    tmp[inds] = ir.deltays_smooth
                    f.create_variable('Civ{}_V_smooth'.format(iuvmat),
                                      (str_nb_vec,), data=tmp)
                    f.create_variable(
                        'Civ{}_U_tps'.format(iuvmat),
                        (str_nb_subdom, 'nb_vec_tps_{}'.format(iuvmat)),
                        data=ir.deltaxs_tps)
                    f.create_variable(
                        'Civ{}_V_tps'.format(iuvmat),
                        (str_nb_subdom,
                         'nb_vec_tps_{}'.format(iuvmat)),
                        data=ir.deltays_tps)
                    tmp = [None] * f.dimensions[str_nb_subdom]
                    for j in range(f.dimensions[str_nb_subdom]):
                        tmp[j] = np.shape(ir.deltaxs_tps[j])[0]
                    f.create_variable(
                        'Civ{}_NbCentres'.format(iuvmat),
                        ('nb_subdomain_{}'.format(iuvmat),),
                        data=tmp)
                except:
                    print('no tps field at passe n {}'.format(iuvmat))

            f.create_variable('Civ{}_C'.format(iuvmat), (str_nb_vec,),
                              data=ir.correls_max)
            tmp = np.zeros(ir.deltaxs.shape).astype('float32')
            indsnan = np.where(np.isnan(ir.deltaxs))
            tmp[indsnan] = 1
            f.create_variable('Civ{}_FF'.format(iuvmat), (str_nb_vec,),
                              data=tmp)

            # mettre bonne valeur de F correspondant a self.piv0.error...
            f.create_variable('Civ{}_F'.format(iuvmat), (str_nb_vec,),
                              data=tmp)

            ## ADD
            # f.create_variable('Civ1_Coord_tps',
            #                   ('nb_subdomain_1', 'nb_coord', 'nb_tps_1'),
            #                   data=???)

        # ADD attributes

    def make_light_result(self, ind_pass=-1):
        piv = self.passes[ind_pass]
        return LightPIVResults(
            piv.deltaxs_approx, piv.deltays_approx,
            piv.ixvecs_grid, piv.iyvecs_grid,
            couple=piv.couple,
            params=piv.params)


class LightPIVResults(DataObject):

    def __init__(self, deltaxs_approx=None, deltays_approx=None,
                 ixvecs_grid=None, iyvecs_grid=None,
                 correls_max=None, correls=None,
                 couple=None, params=None,
                 str_path=None, hdf5_object=None):

        self._keys_to_be_saved = ['xs', 'ys', 'deltaxs', 'deltays']

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

        self.deltaxs = deltaxs_approx
        self.deltays = deltays_approx
        self.couple = couple
        self.params = params
        self.xs = ixvecs_grid
        self.ys = iyvecs_grid

    def _get_name(self):

        serie = self.couple.serie

        str_ind0 = serie._compute_strindices_from_indices(
            *[inds[0] for inds in serie.get_index_slices()])

        str_ind1 = serie._compute_strindices_from_indices(
            *[inds[1] - 1 for inds in serie.get_index_slices()])

        name = ('piv_' + serie.base_name + str_ind0 + '-' + str_ind1 +
                '_light.h5')
        return name

    def save(self, path=None, out_format='uvmat'):

        name = self._get_name()

        if path is not None:
            path_file = os.path.join(path, name)
        else:
            path_file = name

        with h5py.File(path_file, 'w') as f:
            f.attrs['class_name'] = 'LightPIVResults'
            f.attrs['module_name'] = 'fluidimage.data_objects.piv'

            self._save_in_hdf5_object(f, tag='piv')

        return self

    def _save_in_hdf5_object(self, f, tag='piv'):

        if 'class_name' not in f.attrs.keys():
            f.attrs['class_name'] = 'LightPIVResults'
            f.attrs['module_name'] = 'fluidimage.data_objects.piv'
        if 'params' not in f.keys():
            self.params._save_as_hdf5(hdf5_parent=f)
        if 'couple' not in f.keys():
            self.couple.save(hdf5_parent=f)

        g_piv = f.create_group(tag)
        g_piv.attrs['class_name'] = 'LightPIVResults'
        g_piv.attrs['module_name'] = 'fluidimage.data_objects.piv'

        for k in self._keys_to_be_saved:
            g_piv.create_dataset(k, data=self.__dict__[k])

    def _load(self, path):
        with h5py.File(path, 'r') as f:
            self.params = ParamContainer(hdf5_object=f['params'])
            self.couple = ArrayCouple(hdf5_parent=f)

        with h5py.File(path, 'r') as f:
            self._load_from_hdf5_object(f['piv'])

    def _load_from_hdf5_object(self, g_piv):

        f = g_piv.parent

        if not hasattr(self, 'params'):
            self.params = ParamContainer(hdf5_object=f['params'])

        if not hasattr(self, 'couple'):
            self.couple = ArrayCouple(hdf5_parent=f)

        for k in self._keys_to_be_saved:
            dataset = g_piv[k]
            self.__dict__[k] = dataset[:]
