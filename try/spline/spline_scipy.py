
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy

# from scipy.interpolate import SmoothBivariateSpline
from scipy.interpolate import LSQBivariateSpline

xs = np.linspace(-10, 10, 100)
ys = np.linspace(-10, 10, 200)

xknots = yknots = np.linspace(-8, 8, 10)

Xs, Ys = np.meshgrid(xs, ys)

data = Xs**3 + 100*Ys*Xs

data += 0.1 * np.std(data) * scipy.randn(*data.shape)

spline = LSQBivariateSpline(
    Xs.ravel(), Xs.ravel(), data.ravel(), xknots, yknots)

fit = spline(Xs.ravel(), Xs.ravel(), grid=False).reshape(Xs.shape)


fig = plt.Figure()
ax = plt.gca(projection='3d')

ax.plot_surface(Xs, Ys, data)

ax.plot_surface(Xs, Ys, fit, color='r')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
