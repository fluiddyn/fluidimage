"""PIV post-processing
======================


"""

from __future__ import print_function, division

from numbers import Number
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter


class PIV2d(object):
    def __init__(self, x, y, z, vx, vy, vz=np.nan,
                 namevx='vx', namevy='vy', namevz='vz',
                 unitvx='?', unitvy='?', unitvz='?',
                 namex='x', namey='y', namez='z',
                 unitx='?', unity='?', unitz='?'):

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.unitvx = unitvx
        self.namevx = namevx
        self.unitvy = unitvy
        self.namevy = namevy
        self.unitvz = unitvz
        self.namevz = namevz
        self.unitx = unitx
        self.namex = namex
        self.unity = unity
        self.namey = namey
        self.unitz = unitz
        self.namez = namez

        self.name = 'Fluidimage_field'
        self.history = 'fluidimage'
        self.setname = '??'
        self.ysign = 'ysign'

    def __add__(self, other):
        if isinstance(other, Number):
            vx = self.vx + other
            vy = self.vy + other
            vz = self.vz + other
        else:
            vx = self.vx + other.vx
            vy = self.vy + other.vy
            vz = self.vz + other.vz

        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Number):
            vx = self.vx - other
            vy = self.vy - other
            vz = self.vz - other
        else:
            vx = self.vx - other.vx
            vy = self.vy - other.vy
            vz = self.vz - other.vz

        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def __mul__(self, other):
        if isinstance(other, Number):
            vx = other * self.vx
            vy = other * self.vy
            vz = other * self.vz
        else:
            vx = other.vx * self.vx
            vy = other.vy * self.vy
            vz = other.vz * self.vz
        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Number):
            vx = self.vx / other
            vy = self.vy / other
            vz = self.vz / other
        else:
            vx = self.vx / other.vx
            vy = self.vy / other.vy
            vz = self.vz / other.vz
        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def display(self, scale=1, background=None):
        fig = plt.figure()
        ax = plt.gca()

        pcm = ax.pcolormesh(self.x, self.y, self.compute_norm())

        q = ax.quiver(self.x, self.y, self.vx, self.vy,
                      scale_units='xy', scale=scale)
        ax.set_xlabel(self.namex + ' [' + self.unitx + ']')
        ax.set_ylabel(self.namey + ' [' + self.unity + ']')

        def onclick(event):
            key = event.key
            if key == 'ctrl++':
                q.scale *= 2.
                print(key + ': multiply q.scale by 2.', end='')
            elif key == 'ctrl+-':
                q.scale /= 2.
                print(key + ': divide q.scale by 2.', end='')
            if event.key in ['ctrl++', 'ctrl+-']:
                print(' q.scale = {}'.format(q.scale))
                fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', onclick)
        plt.colorbar(pcm, ax=ax)

        xmin = self.x.min()
        xmax = self.x.max()
        lx = xmax - xmin
        ymin = self.y.min()
        ymax = self.y.max()
        ly = ymax - ymin
        n = 20
        ax.set_xlim([xmin - lx/n, xmax + lx/n])
        ax.set_ylim([ymin - ly/n, ymax + ly/n])
        ax.set_aspect('equal')
        plt.show()

    def median_filter(self, size, niter=1, valid=True):

        def _medianf(f):
            for i in range(niter):
                f = median_filter(f, size)
            return f

        ret = deepcopy(self)
        ret.vx = _medianf(self.vx)
        ret.vy = _medianf(self.vy)

        if hasattr(self, 'vz'):
            ret.vz = _medianf(self.vz)

        if valid:
            mf = int(np.floor(max(size)/2))
            ny, nx = self.vx.shape
            ret = ret.extract(mf, ny-mf, mf, nx-mf)
        return ret

    def extract(self, start0, stop0, start1, stop1):
        def _extract2d(f):
            return f[start0:stop0, start1:stop1]
        ret = deepcopy(self)
        ret.x = ret.x[start1:stop1]
        ret.y = ret.y[start0:stop0]
        ret.vx = _extract2d(ret.vx)
        ret.vy = _extract2d(ret.vy)
        if hasattr(self, 'vz'):
            ret.vz = _extract2d(ret.vz)
        return ret

    def truncate(self, cut=0, phys=False):
        if phys:
            raise NotImplementedError
        ny, nx = self.vx.shape
        return self.extract(cut, ny-cut, cut, nx-cut)

    def compute_norm(self):
        return np.sqrt(self.vx**2 + self.vy**2)

function_names = ['median_filter', 'extract']


class ArrayPIV(object):
    def __init__(self, l=None):
        if l is None:
            l = []
        elif not isinstance(l, list):
            raise TypeError
        self._list = l

    def append(self, v):
        self._list.append(v)

    def extend(self, l):
        self._list.extend(l)

    def __add__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(v + other)
        return result

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(v * other)
        return result

    def __rmul__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(other * v)
        return result

    def __div__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(v / other)
        return result

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, key):
        r = self._list.__getitem__(key)
        if isinstance(r, list):
            r = type(self)(r)
        return r

    def __setitem__(self, key, value):
        self._list.__setitem__(key, value)

    def __delitem__(self, key):
        self._list.__detitem__(key)

    def __repr__(self):
        return 'ArrayPIV containing {} fields:\n'.format(len(self)) + \
            self._list.__repr__()

    def __len__(self):
        return self._list.__len__()

    def median_filter(self, size, niter=1, valid=True):
        result = type(self)()
        for v in self:
            result.append(v.median_filter(size, niter=niter, valid=valid))
        return result

    def extract(self, start0, stop0, start1, stop1):
        result = type(self)()
        for v in self:
            result.append(v.extract(start0, stop0, start1, stop1))
        return result

    def truncate(self, cut=0, phys=False):
        result = type(self)()
        for v in self:
            result.append(v.truncate(cut=cut, phys=phys))
        return result
