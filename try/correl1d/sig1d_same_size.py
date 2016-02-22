
import numpy as np

import matplotlib.pyplot as plt

from numpy.fft import fft, ifft

from scipy.signal import correlate
# 2d equivalent: from scipy.signal import correlate2d

nb_points = 64
xs = np.arange(nb_points) - nb_points/2.

nb_particles = nb_points // 3
xmax = xs.max() - 10
xmin = xs.min() + 10
xparts = [xmin + (xmax - xmin) * np.random.rand() for i in range(nb_particles)]


def f(x):
    result = np.zeros_like(x)

    for xpart in xparts:
        result += np.exp(-0.5*(x-xpart)**2)

    return result

epsilon = 0.
displacement = 20.
in0 = f(xs) + epsilon * np.random.randn(*xs.shape)
in1 = f(xs - displacement) + epsilon * np.random.randn(*xs.shape)

norm = np.sum(in1**2)

c_full = correlate(in0, in1, mode='full')/norm
c_same = correlate(in0, in1, mode='same')/norm
c_valid = correlate(in0, in1, mode='valid')/norm

c_fft = ((ifft(fft(in0).conj() * fft(in1))).real)[::-1]/norm


if not int(displacement) == c_full.shape[0]//2 - c_full.argmax():
    print('We do not understand (?)')

if not int(displacement) == c_same.shape[0]//2 - c_same.argmax():
    print('We do not understand (?)')

if not c_fft.argmax() == c_full.argmax():
    print('We do not understand (?)')

plt.figure()
ax = plt.gca()

ax.plot(xs, in0, 'b')
ax.plot(xs, in1, 'r')


plt.figure()
ax = plt.gca()

ax.plot(c_full, 'b')
ax.plot(c_valid, 'r')
ax.plot(c_same, 'g')
ax.plot(c_fft, 'y')

plt.show()
