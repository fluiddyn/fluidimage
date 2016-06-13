
import os
from runpy import run_path

from setuptools import setup, find_packages

try:
    from pythran.dist import PythranExtension
    use_pythran = True
except ImportError:
    use_pythran = False

# Get the long description from the relevant file
with open('README.rst') as f:
    long_description = f.read()
lines = long_description.splitlines(True)
long_description = ''.join(lines[13:])


print(long_description)

# Get the version from the relevant file
d = run_path('fluidimage/_version.py')
__version__ = d['__version__']

install_requires = ['numpy', 'fluiddyn >= 0.0.12a4']

on_rtd = os.environ.get('READTHEDOCS')
if not on_rtd:
    install_requires.extend(['scipy', 'h5py', 'h5netcdf'])

if use_pythran:
    ext_modules = [
        PythranExtension(
            'fluidimage.calcul.correl_pythran',
            ['fluidimage/calcul/correl_pythran.py']),
        PythranExtension(
            'fluidimage.calcul.interpolate.tps_pythran',
            ['fluidimage/calcul/interpolate/tps_pythran.py'])]
else:
    ext_modules = []


setup(
    name='fluidimage',
    version=__version__,
    description=('fluid image processing with Python.'),
    long_description=long_description,
    keywords='PIV',
    author='Pierre Augier',
    author_email='pierre.augier@legi.cnrs.fr',
    url='https://bitbucket.org/fluiddyn/fluidimage',
    license='CeCILL',
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        # actually CeCILL License (GPL compatible license for French laws)
        #
        # Specify the Python versions you support here. In particular,
        # ensure that you indicate whether you support Python 2,
        # Python 3 or both.
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Cython',
        'Programming Language :: C'],
    packages=find_packages(exclude=[
        'doc', 'include', 'scripts']),
    install_requires=install_requires,
    ext_modules=ext_modules)
