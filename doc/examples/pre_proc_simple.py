
from __future__ import print_function

# import matplotlib.pyplot as plt

from fluidimage.pre_proc.base import PreprocBase


params = PreprocBase.create_default_params()

params.preproc.series.path = '../../image_samples/Karman/Images'
print('Available preprocessing tools: ', params.preproc.tools.available_tools)

params.preproc.tools.sequence = ['sliding_median', 'global_threshold']
params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.window_size = 25

params.preproc.tools.global_threshold.enable = True
params.preproc.tools.global_threshold.minima = 0.
params.preproc.tools.global_threshold.maxima = 255.

preproc = PreprocBase(params)
preproc()

#def plot(nimages=None):
#    after = preproc.results
#    before = {}
#    name_files = preproc.serie_arrays.get_name_files()[:nimages]
#    for fname in name_files:
#        before[fname] = preproc.serie_arrays.get_array_from_name(fname)
#
#    if nimages is None:
#        nimages = len(name_file)s
#
#    fig, axes = plt.subplots(ncols=nimages, nrows=2)
#    fig.tight_layout(h_pad=0.001, w_pad=0.001)
#    ax = axes.ravel()
#    for n, fname in enumerate(name_files):
#        ax[2 * n].imshow(before[fname], cmap='gray', aspect='auto')
#        ax[2 * n].set_title('Before: ' + fname, size='small')
#
#        ax[2 * n + 1].imshow(after[fname], cmap='gray', aspect='auto')
#        ax[2 * n + 1].set_title('After: ' + fname, size='small')
#        ax[2 * n + 1].axis('off')
#
#    plt.show()
#
#
#prompt = raw_input('Plot results [Y/n]? ')
#nimages = raw_input('Number of images to plot [default=1]? ')
#if(prompt == '' or prompt == 'y' or prompt == 'Y'):
#    if nimages == '':
#        plot(1)
#    else:
#        plot(int(nimages))


preproc.display(1, hist=False)
