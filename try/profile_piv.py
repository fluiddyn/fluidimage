
import pstats
import cProfile

from copy import copy


from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays

from base import PIVSerie

path = '/fsnet/project/meige/2016/16FLUIDIMAGE/samples/Oseen'
base_name = 'PIVlab_Oseen_z'


def give_indslices_from_indserie(iserie):
    indslices = copy(serie_arrays._index_slices_all_files)
    indslices[0] = [2*iserie+1, 2*iserie+3, 1]
    return indslices

serie_arrays = SerieOfArraysFromFiles(path, base_name=base_name)
series = SeriesOfArrays(serie_arrays, give_indslices_from_indserie,
                        ind_stop=None)

o = PIVSerie(
    series_images=series, n_interrogation_window=64, overlap=0.9)

cProfile.runctx('o.compute_outputs()',
                globals(), locals(), 'Profile.prof')

s = pstats.Stats('Profile.prof')
s.strip_dirs().sort_stats('time').print_stats(10)
