
import numpy as np


# pythran export compute_tps_matrix(float64[][], float64[][])


def compute_tps_matrix(new_pos, centers):
    """calculate the thin plate spline (tps) interpolation at a set of points

    Parameters
    ----------

    dsites: np.array
        ``[nb_dim, M]`` array representing the postions of the M
        'observation' sites, with nb_dim the space dimension.

    centers: np.array
        ``[nb_dim, N]`` array representing the postions of the N centers,
        sources of the tps.

    Returns
    -------

    EM : np.array
        ``[(N+nb_dim), M]`` matrix representing the contributions at the M sites.

        From unit sources located at each of the N centers, +
        (nb_dim+1) columns representing the contribution of the linear
        gradient part.

    Notes
    -----

    >>> U_interp = np.dot(U_tps, EM)

    """

    d, nb_new_pos = new_pos.shape
    d2, nb_centers = centers.shape
    assert d == d2

    EM = np.zeros((nb_centers, nb_new_pos))

    # # pythran 0.7.4 does not know np.meshgrid
    # for ind_d in range(s):
    #     Dsites, Centers = np.meshgrid(dsites[ind_d], centers[ind_d])
    #     EM += (Dsites - Centers)**2

    for ind_d in range(d):
        for ic, center in enumerate(centers[ind_d]):
            for inp, npos in enumerate(new_pos[ind_d]):
                EM[ic, inp] += (npos - center) ** 2

    # Pythran does not like that!
    # nb_p = np.where(EM != 0)
    # EM[nb_p] = EM[nb_p] * np.log(EM[nb_p]) / 2

    for ic in range(nb_centers):
        for inp in range(nb_new_pos):
            tmp = EM[ic, inp]
            if tmp != 0:
                EM[ic, inp] = tmp * np.log(tmp) / 2

    # # pythran 0.7.4 does not know np.vstack
    # EM_ret = np.vstack([EM, np.ones(M), dsites])

    EM_ret = np.empty((nb_centers + 1 + d, nb_new_pos))
    EM_ret[:nb_centers, :] = EM
    EM_ret[nb_centers, :] = np.ones(nb_new_pos)
    EM_ret[nb_centers + 1 : nb_centers + 1 + d, :] = new_pos

    return EM_ret
