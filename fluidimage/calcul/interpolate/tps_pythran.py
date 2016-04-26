
import numpy as np


# pythran export compute_tps_matrix_pythran(float64[][], float64[][])

def compute_tps_matrix_pythran(dsites, centers):
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

    d, M = dsites.shape
    d2, N = centers.shape
    assert d == d2

    EM = np.zeros([N, M])

    for ind_d in range(d):
        # for ids, dsite in enumerate(dsites[ind_d, :]):
        #     for ic, center in enumerate(centers[ind_d, :]):
        #         EM[ids, ic] = EM[ids, ic] + (dsite - center)**2
        for ids in range(N):
            for ic in range(M):
                EM[ids, ic] = EM[ids, ic] + (
                    dsites[ind_d, ids] - centers[ind_d, ic])**2

                
    # # pythran 0.7.4 does not know np.meshgrid
    # for ind_d in range(s):
    #     Dsites, Centers = np.meshgrid(dsites[ind_d], centers[ind_d])
    #     EM += (Dsites - Centers)**2

    # for ids in range(N):
    #     for ic in range(M):
    #         tmp = EM[ids, ic]
    #         if tmp != 0:
    #             EM[ids, ic] = tmp * np.log(tmp) / 2

    # nb_p = np.where(EM != 0)
    # EM[nb_p] = EM[nb_p] * np.log(EM[nb_p]) / 2

    # EM_ret = EM
    
    # EM_ret = np.empty([N + 1 + d, M])

    # EM_ret[:N, :] = EM
    # EM_ret[N, :] = np.ones(M)
    # EM_ret[N+1: N+1+d, :] = dsites

    # # pythran 0.7.4 does not know np.vstack
    # EM_ret = np.vstack([EM, np.ones(M), dsites])
    return 1
