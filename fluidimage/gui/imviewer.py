
from __future__ import print_function, division

import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.widgets import TextBox, Button


def check_image(path):
    return path[-3:] in ['png']


size_button = 0.06
sep_button = 0.02
x0 = 0.3

name_buttons = ['-n', '-1', '+1', '+n']

x_buttons = [x0 + i * (size_button + sep_button)
             for i in range(len(name_buttons))]

print(x_buttons)


def _create_button(fig, rect, text, func):
    ax = fig.add_axes(rect)
    button = Button(ax, text)
    button.on_clicked(func)
    return button


def _create_text(fig, rect, name, func, initial):
    ax = fig.add_axes(rect)
    textbox = TextBox(ax, name, initial=initial)
    textbox.on_submit(func)
    return textbox


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='Path file or directory.', type=str,
        nargs='?', default=os.getcwd())
    parser.add_argument(
        '-cm', '--colormap', help='colormap', type=str, default='')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='count')

    return parser.parse_args()


class ImageViewer(object):

    def __init__(self, args):

        path_in = args.path
        if os.path.isdir(path_in):
            self.path_files = glob(os.path.join(path_in, '*'))
            self.path_files = [path for path in self.path_files
                               if check_image(path)]
            self.path_files.sort()
            ifile = 0
        else:
            path_file = glob(path_in)[0]
            self.path_files = glob(os.path.join(
                os.path.split(path_file)[0], '*'))
            self.path_files = [path for path in self.path_files
                               if check_image(path)]
            self.path_files.sort()
            ifile = self.path_files.index(path_file)

        # print(self.path_files)

        fig = self.fig = plt.figure()
        self.ax = fig.add_axes([0.06, 0.2, 0.7, 0.75])

        self.maps = {}
        self.cmap = plt.cm.jet

        self.ifile = ifile
        self._last_was_increase = False
        im = imread(self.path_files[ifile])

        self.clim = [0, 0.99*im.max()]

        self.mappable = self.ax.imshow(
            im, interpolation='nearest', cmap=self.cmap, origin='upper',
            extent=[0, im.shape[1], im.shape[0], 0],
            vmin=self.clim[0], vmax=self.clim[1])

        self.maps[ifile] = self.mappable
        self._image_changing = False

        function_buttons = [self._do_nothing] * len(name_buttons)
        function_buttons[0] = self._decrease_ifile_n
        function_buttons[1] = self._decrease_ifile
        function_buttons[2] = self._increase_ifile
        function_buttons[3] = self._increase_ifile_n
        self._buttons = []
        y = size_button
        for i, x in enumerate(x_buttons):
            name = name_buttons[i]
            func = function_buttons[i]
            self._buttons.append(
                _create_button(
                    fig, [x, y, size_button, size_button], name, func))

        self._n = 1

        self.textbox = _create_text(
            fig, [0.1, y, 2*size_button, size_button],
            'n = ', self._submit_n, '1')

        self.textboxmax = _create_text(
            fig, [0.85, 0.92, 2*size_button, size_button],
            'n = ', self._change_cmax, '{:.4f}'.format(self.clim[1]))

        self.textboxmin = _create_text(
            fig, [0.85, 0.1, 2*size_button, size_button],
            'n = ', self._change_cmin, '{:.4f}'.format(self.clim[0]))

        cax = fig.add_axes([0.82, 0.2, 0.07, 0.7])
        self.cbar = fig.colorbar(self.mappable, cax=cax)

        fig.canvas.mpl_connect('key_press_event', self.onclick)
        print('press alt+h for help')

        plt.show()

    def change_im(self):
        if self._image_changing:
            return
        self._image_changing = True
        ifile = self.ifile
        name_file = os.path.split(self.path_files[ifile])[-1]
        print('changing to file ' + name_file, end='...')
        map_old = self.mappable

        if ifile in self.maps:
            self.mappable = self.maps[ifile]
            self.mappable.set_visible(True)
        else:
            im = imread(self.path_files[ifile])
            self.mappable = self.ax.imshow(
                im, interpolation='nearest', cmap=self.cmap, origin='upper',
                extent=[0, im.shape[1], im.shape[0], 0],
                vmin=self.clim[0], vmax=self.clim[1])
            self.maps[i] = self.mappable

        map_old.set_visible(False)
        self.ax.set_title(name_file)
        self.fig.canvas.draw()
        print('\rchanged to file ' + name_file + ' ' * 20)
        self._image_changing = False

    def _switch(self):
        if self._last_was_increase:
            self._decrease_ifile_n()
        else:
            self._increase_ifile_n()
        self._last_was_increase = not self._last_was_increase

    def _do_nothing(self, event):
        print('do nothing')

    def _increase_ifile(self, event=None):
        self.ifile += 1
        self.change_im()

    def _decrease_ifile(self, event=None):
        self.ifile -= 1
        self.change_im()

    def _increase_ifile_n(self, event=None):
        self.ifile += self._n
        self.change_im()

    def _decrease_ifile_n(self, event=None):
        self.ifile -= self._n
        self.change_im()

    def _submit_n(self, text):
        self._n = eval(text, {}, {})
        print('n = ', self._n)

    def _change_cmax(self, text):
        cmax = eval(text, {}, {})
        self.clim[1] = cmax
        self._update_clim()

    def _change_cmin(self, text):
        cmin = eval(text, {}, {})
        self.clim[0] = cmin
        self._update_clim()

    def _update_clim(self):
        self.mappable.set_clim(self.clim)
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.key == 'alt+h':
            print('\nclick on a vector to show information\n'
                  'alt+s\t\t\t switch between images\n'
                  'alt+left or alt+right\t change vector.')

        if event.inaxes != self.ax:
            return

        if event.key == 'alt+s':
            self._switch()

if __name__ == '__main__':
    args = parse_args()
    self = ImageViewer(args)
