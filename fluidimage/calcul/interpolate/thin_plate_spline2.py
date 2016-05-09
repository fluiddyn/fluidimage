"""Thin plate spline
====================

Translated and adapted from UVmat code (Joel Sommeria, LEGI, CNRS).

This interpolation/smoothing (Duchon, 1976; NguyenDuc and Sommeria,
1988) minimises a linear combination of the squared curvature and
squared difference from the initial data.

We first need to compute tps coefficients ``U_tps`` (function
``compute_tps_coeff``). Interpolated data can then be obtained as the
matrix product ``dot(U_tps, EM)`` where the matrix ``EM`` is obtained
by the function ``compute_tps_matrix``.  The spatial derivatives are
obtained as ``dot(U_tps, EMDX)`` and ``dot(U_tps, EMDY)``, where
``EMDX`` and ``EMDY`` are obtained from the function
``compute_tps_matrix_dxy``. A helper class is also provided.

.. autofunction:: compute_tps_coeff

.. autoclass:: ThinPlateSpline
   :members:

.. autofunction:: compute_tps_matrix

.. autofunction:: compute_tps_matrices_dxy


.. todo::

   Understand and do as UVmat (TPS with subdomains).

"""
import numpy as np

class ThinPlateSpline(object):
    """Helper class for thin plate interpolation."""

    def __init__(self, new_positions, centers, U, subdom_size, smoothing_coef, 
                 threshold=None, pourc_buffer_area=0.2):
        
        self.centers = centers
        self.subdom_size = subdom_size
        self.new_positions = new_positions
        self.smoothing_coef = smoothing_coef
        self.threshold = threshold
        
        self.U = U
        
        self.compute_indices(pourc_buffer_area)
        
        self.compute_tps_coeff_subdom()
        
        self.compute_tps_matrices_dxy()
        
        #self.EM = compute_tps_matrix(new_positions, centers)
        #self.DMX, self.DMY = compute_tps_matrices_dxy(new_positions, centers)

    def compute_field(self):
        """Compute the interpolated field."""
        return np.dot(self.U_tps, self.EM)

    def compute_gradient(self, U_tps):
        """Compute the gradient (dx_U, dy_U)"""
        return np.dot(U_tps, self.DMX), np.dot(U_tps, self.DMY)


    def compute_indices(self, pourc_buffer_area=0.2):
        max_coord = np.max(self.centers, 1)
        min_coord = np.min(self.centers, 1)
        range_coord = max_coord - min_coord
        aspect_ratio = range_coord[1] / range_coord[0]
    
        nb_subdom = self.centers[0,:].size / self.subdom_size
        nb_subdomx = int( np.max ( np.floor( np.sqrt( nb_subdom / aspect_ratio ) ), 0) )
        nb_subdomy = int( np.max ( np.floor( np.sqrt( nb_subdom * aspect_ratio ) ), 0) )
        nb_subdom = nb_subdomx * nb_subdomy
    
        x_dom = np.arange(min_coord[0],max_coord[0],range_coord[0] / nb_subdomx)
        x_dom = np.unique( np.append(x_dom, max_coord[0]) )
        y_dom = np.arange(min_coord[1],max_coord[1],range_coord[1] / nb_subdomy)
        y_dom = np.unique( np.append(y_dom, max_coord[1]) )

        buffer_area_x = x_dom*0 + range_coord[0]/(nb_subdomx) * pourc_buffer_area
        buffer_area_y = y_dom*0 + range_coord[1]/(nb_subdomy) * pourc_buffer_area

        ind_subdom = np.zeros([nb_subdom,2])    
        ind_v_subdom = []               
        if self.new_positions is not None:
            ind_new_positions_subdom= []
        
        count = 0
        for i in range(nb_subdomx):
            for j in range(nb_subdomy):
                ind_subdom[count,:] = [i, j]
            
                ind_v_subdom.append( np.argwhere( 
                (self.centers[0 , :] >= x_dom[i] - buffer_area_x[i]) & 
                (self.centers[0 , :] < x_dom[i+1] + buffer_area_x[i+1]) & 
                (self.centers[1 , :] >= y_dom[j] - buffer_area_y[j]) & 
                (self.centers[1 , :] < y_dom[j+1] + buffer_area_y[j+1]) ).flatten())
            
                if self.new_positions is not None:
                    ind_new_positions_subdom.append( np.argwhere( 
                    (self.new_positions[0 , :] >= x_dom[i] - buffer_area_x[i]) & 
                    (self.new_positions[0 , :] < x_dom[i+1] + buffer_area_x[i+1]) & 
                    (self.new_positions[1 , :] >= y_dom[j] - buffer_area_y[j]) & 
                    (self.new_positions[1 , :] < y_dom[j+1] + buffer_area_y[j+1]) ).flatten())        
            
                count +=1
        self.ind_v_subdom = ind_v_subdom
        self.ind_new_positions_subdom = ind_new_positions_subdom
        self.nb_subdom = nb_subdom
    
    

    def compute_tps_coeff_subdom(self):
        
        U_tps = [None] * self.nb_subdom
        U_smooth = [None] * self.nb_subdom
        EM = [None] * self.nb_subdom

        for i in range(self.nb_subdom):
            centerstemp=np.vstack( [self.centers[0][self.ind_v_subdom[i]], self.centers[1][self.ind_v_subdom[i]] ] )
            Utemp = self.U[self.ind_v_subdom[i]]
            U_smooth[i], U_tps[i] = self.compute_tps_coeff_iter(centerstemp, Utemp)
        
            centers_newposition_temp=np.vstack( [self.new_positions[0][self.ind_new_positions_subdom[i]], 
                                                 self.new_positions[1][self.ind_new_positions_subdom[i]] ] )
        
            EM[i] = self.compute_tps_matrix(centers_newposition_temp, centerstemp)
        
        self.U_smooth = U_smooth 
        self.U_tps = U_tps
        self.EM = EM 
    
    

    def compute_tps_coeff_iter(self, centers, U):
        """ Compute the thin plate spline (tps) coefficients removing erratic 
        vectors
        It computes iteratively "compute_tps_coeff", compares the tps result 
        to the initial data and remove it if difference is larger than the given 
        threshold

        """
        U_smooth, U_tps = self.compute_tps_coeff(centers, U)
        if self.threshold is not None:
            Udiff = np.sqrt( (U_smooth - U)**2 )
            ind_erratic_vector = np.argwhere(Udiff > self.threshold)
        
            count = 1
            while ind_erratic_vector.size != 0:
                U[ind_erratic_vector] = U_smooth[ind_erratic_vector]
                U_smooth, U_tps = self.compute_tps_coeff(centers, U)  
            
                Udiff = np.sqrt( (U_smooth-U)**2 )
                ind_erratic_vector = np.argwhere(Udiff > self.threshold)
                count += 1
            
                if count > 10:
                    print('tps stopped after 10 iterations')                
                    break
        if count > 1  :      
            print ('tps done after ', count, ' attempt(s)') 
        return U_smooth, U_tps
        
    

    def compute_tps_coeff(self, centers, U):
        """Calculate the thin plate spline (tps) coefficients

        Parameters
        ----------

        centers : np.array
            ``[nb_dim,  N]`` array representing the positions of the N centers,
            sources of the TPS (nb_dim = space dimension).

        U : np.array
            ``[N]`` array representing the values of the considered
            scalar measured at the centres ``centers``.

        smoothing_coef : float
            Smoothing parameter. The result is smoother for larger smoothing_coef.

        Returns
        -------

        U_smooth : np.array
             Values of the quantity U at the N centres after smoothing.

        U_tps : np.array
             TPS weights of the centres and columns of the linear.

        """
        nb_dim, N = centers.shape
        U = np.hstack([U, np.zeros(nb_dim + 1)])
        U = U.reshape([U.size, 1])
        EM = self.compute_tps_matrix(centers, centers).T
        smoothing_mat = self.smoothing_coef * np.eye(N, N)
        smoothing_mat = np.hstack([smoothing_mat, np.zeros([N, nb_dim + 1])])
        PM = np.hstack([np.ones([N, 1]), centers.T])
        IM = np.vstack([EM + smoothing_mat,
                        np.hstack([PM.T, np.zeros([nb_dim + 1, nb_dim + 1])])])
        U_tps, r, r2, r3 = np.linalg.lstsq(IM, U)
        U_smooth = np.dot(EM, U_tps)
        return U_smooth.ravel(), U_tps.ravel()
        
    def compute_tps_matrix(self, dsites, centers):
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
        s, M = dsites.shape
        s2, N = centers.shape
        assert s == s2
        EM = np.zeros([N, M])
        for d in range(s):
            Dsites, Centers = np.meshgrid(dsites[d], centers[d])
            EM = EM + (Dsites - Centers) ** 2
                
        nb_p = np.where(EM != 0)
        EM[nb_p] = EM[nb_p] * np.log(EM[nb_p]) / 2
        EM = np.vstack([EM, np.ones(M), dsites])
        return EM
        
    def compute_tps_matrices_dxy(self):
        """Calculate the derivatives of thin plate spline (tps) interpolation
        at a set of points (limited to the 2D case)

        Parameters
        ----------

        dsites : np.array
            ``[nb_dim,  M]`` array of interpolation site coordinates
            (nb_dim = space dimension = 2, here).

        centers : np.array
            ``[nb_dim,  N]`` array of centre coordinates (initial data).

        Returns
        -------

        DMX : np.array
            ``[(N+3),  M]`` array representing the contributions to the X
            derivatives at the M sites from unit sources located at each
            of the N centers, + 3 columns representing the contribution of
            the linear gradient part.

        DMY : np.array
            idem for Y derivatives.

        """

        DMX_tps = [None] * self.nb_subdom            
        DMY_tps = [None] * self.nb_subdom
        DUX_smooth = [None] * self.nb_subdom            
        DUY_smooth = [None] * self.nb_subdom
        
        
        for i in range(self.nb_subdom):
            centers_newposition_temp=np.vstack( [self.new_positions[0][self.ind_new_positions_subdom[i]], 
                                                 self.new_positions[1][self.ind_new_positions_subdom[i]] ] )
            centers_temp=np.vstack( [self.centers[0][self.ind_v_subdom[i]], 
                                                 self.centers[1][self.ind_v_subdom[i]] ] )
            s, M = centers_newposition_temp.shape
            s2, N = centers_temp.shape
            assert s == s2
            Dsites, Centers = np.meshgrid(centers_newposition_temp[0], centers_temp[0])
            DX = Dsites - Centers
            Dsites, Centers = np.meshgrid(centers_newposition_temp[1], centers_temp[1])
            DY = Dsites - Centers
            DM = DX * DX + DY * DY
            DM[DM != 0] = np.log(DM[DM != 0]) + 1
            
            DMX_tps[i] = np.vstack([DX * DM, np.zeros(M), np.ones(M), np.zeros(M)])
            DMY_tps[i] = np.vstack([DY * DM, np.zeros(M), np.zeros(M), np.ones(M)])
            DUX_smooth[i] = np.dot(self.U_tps[i], DMX_tps[i])
            DUY_smooth[i] = np.dot(self.U_tps[i], DMY_tps[i])
            
            
                
        self.DMX_tps =DMX_tps
        self.DMY_tps = DMY_tps
        self.DUX_smooth = DUX_smooth        
        self.DUY_smooth = DUY_smooth

        #return DMX, DMY
        
        
    def compute_U_eval(self):
        
        U_eval = np.zeros(self.new_positions[0].shape)
        nb_tps = np.zeros(self.new_positions[0].shape)
        
        for i in range(self.nb_subdom):
            U_eval[self.ind_new_positions_subdom[i]] += np.dot(self.U_tps[i], self.EM[i])

            nb_tps[self.ind_new_positions_subdom[i]] += 1.0

        U_eval /= nb_tps

        self.U_eval = U_eval
        return U_eval
        
    def compute_dxy_eval(self):
        
        DMX_eval = np.zeros(self.new_positions[0].shape)        
        DMY_eval = np.zeros(self.new_positions[0].shape)
        nb_tps = np.zeros(self.new_positions[0].shape)
        
        for i in range(self.nb_subdom):
            DMX_eval[self.ind_new_positions_subdom[i]] += np.dot(self.U_tps[i], self.DMX_tps[i])
            DMY_eval[self.ind_new_positions_subdom[i]] += np.dot(self.U_tps[i], self.DMY_tps[i])
            nb_tps[self.ind_new_positions_subdom[i]] += 1.0

        DMX_eval /= nb_tps
        DMY_eval /= nb_tps

        self.DMX_eval = DMX_eval
        self.DMY_eval = DMY_eval

        return DMX_eval, DMY_eval
        

