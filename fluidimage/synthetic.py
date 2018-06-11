
import numpy as np


def make_synthetic_images(
    displacements,
    nb_particles,
    shape_im0,
    shape_im1=None,
    epsilon=0.,
    part_size=np.sqrt(1 / 0.5),
):
    ny, nx = tuple(shape_im0)

    displacement_x, displacement_y = tuple(displacements)

    xs = np.arange(nx, dtype="float32")
    ys = np.arange(ny, dtype="float32")

    Xs, Ys = np.meshgrid(xs, ys)

    xmax = xs.max() + abs(displacement_x)
    xmin = xs.min() - abs(displacement_x)
    xparts = [
        xmin + (xmax - xmin) * np.random.rand() for i in range(nb_particles)
    ]
    ymax = ys.max() + abs(displacement_y)
    ymin = ys.min() - abs(displacement_y)
    yparts = [
        ymin + (ymax - ymin) * np.random.rand() for i in range(nb_particles)
    ]

    def f(x, y):
        result = np.zeros_like(x, dtype="float32")

        for xpart, ypart in zip(xparts, yparts):
            result += np.exp(
                -(1 / part_size ** 2) * ((x - xpart) ** 2 + (y - ypart) ** 2)
            )

        return result

    im0 = f(Xs, Ys)
    im1 = f(Xs - displacement_x, Ys - displacement_y)

    if epsilon > 0:
        im0 += epsilon * np.random.randn(*im0.shape)
        im1 += epsilon * np.random.randn(*im1.shape)

    if shape_im1 is not None:
        ny1, nx1 = tuple(shape_im1)
        if ny1 > ny or nx1 > nx:
            raise ValueError("ny1 > ny or nx1 > nx")

        ixfirst = (nx - nx1) // 2
        iyfirst = (ny - ny1) // 2
        im1 = im1[iyfirst : iyfirst + ny1, ixfirst : ixfirst + nx1]

    return im0, im1
