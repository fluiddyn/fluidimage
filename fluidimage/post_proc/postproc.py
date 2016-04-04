import os   
import numpy as np
from fluidimage.data_objects.piv import LightPIVResults
from displayf import displayf
from computation_functions import(compute_grid, compute_derivatives, 
                                  compute_rot, compute_div, compute_ken, 
                                  compute_norm, oneD_fourier_transform, 
                                   twoD_fourier_transform)
import time
import pylab

class DataObject(object):
    pass


class PIV_Postproc(LightPIVResults):

    def __init__(self, path):
        super(PIV_Postproc, self).__init__(str_path=path)
        self.path = os.path.abspath(path)
        self.X, self.Y, self.dx, self.dy, self.U, self.V = self.compute_grid()
        
    def displayf(self, U=None, V=None, X=None, Y=None, bg=None, *args):
        if bg is None:
            bg = self. compute_norm(U, V)
            
        if X is None:
            X = self.X
            Y = self.Y
            
        displayf(X, Y, U=U, V=V, background=bg, *args)
        
    def spatial_average(self, U):
        U_mean = np.mean(U,(1,2))
        return U_mean    
        
    def compute_grid(self):
        X, Y, dx, dy, U, V = compute_grid(self.xs, self.ys, self.deltaxs, self.deltays)
        return X, Y, dx, dy, U, V

    def compute_derivatives(self,edge_order=2):
        dUdx, dUdy, dVdx, dVdy = compute_derivatives(self.dx, self.dy, self.U, self.V, edge_order=2)
        return dUdx, dUdy, dVdx, dVdy
                
    def compute_rot(self, edge_order=2):
        if ~hasattr(self, 'dUdx'):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(edge_order = edge_order)
        rot = compute_rot(self.dUdy, self.dVdx)
        return rot
     
    def compute_div(self, edge_order=2):
        if ~hasattr(self, 'dUdx'):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(edge_order=edge_order)   
        div = compute_div(self.dUdx, self.dVdy)  
        return div
        
    def compute_ken(self):           
        ken = compute_ken(self.U, self.V)  
        return ken    
    
    def compute_norm(self, U, V):
        norm = compute_norm(self.U, self.V)
        return norm
        
    def compute_spatial_fft(self):
        
        fftU, kx, ky, psdU = twoD_fourier_transform(self.X, self.Y, self.U, axis=(1,2), parseval=False)
        fftV, kx, ky, psdV = twoD_fourier_transform(self.X, self.Y, self.V, axis=(1,2), parseval=False)
        
        if not hasattr(self, 'fft'):
            self.fft = DataObject()
        if not hasattr(self.fft, 'spatial'):
            time=DataObject()
            self.fft.spatial = spatial        
        
        self.fft.spatial.kx = kx
        self.fft.spatial.ky = ky        
        self.fft.spatial.fftU = fftU        
        self.fft.spatial.fftV = fftV
        self.fft.spatial.psdU = psdU        
        self.fft.spatial.psdV = psdV   

class PIV_PostProc_serie(LightPIVResults):
    
    def __init__(self, path=None):
        self.path = path
        path0 = path[0]
        super(PIV_PostProc_serie, self).__init__(str_path=path0)
        for ti, pathi in enumerate (path[1:]):
            temp=PIV_Postproc(path=pathi)
            self.deltaxs = np.vstack([self.deltaxs, temp.deltaxs])
            self.deltays = np.vstack([self.deltays, temp.deltays])
        self.X, self.Y, self.dx, self.dy, self.U, self.V = self.compute_grid()
    
    def set_time(self, t):
        self.t = np.linspace(0, np.size(self.path), np.size(self.path))
   
    def displayf(self, U=None, V=None, bg=None, X=None, Y=None, 
                 timesleep=0.5, *args):
        if U is None and bg.ndim == 3:
            U = V =[None]*len(bg)            
        elif bg is None and U.ndim ==3:
            bg = self.compute_norm(U, V)
        
        if X is None:
            X = self.X
            Y = self.Y
        
        if U is None:
            displayf(X, Y, U=U, V=V, background=bg, *args)
        else:
            for i in range(len(U)):
                displayf(X, Y, U=U[i], V=V[i], background=bg[i], *args) 
                pylab.show()
                time.sleep(timesleep)
                
    def time_average(self, U):
        U_mean = np.mean(U,0)
        return U_mean
        
    def spatial_average(self, U):
        U_mean = np.mean(U,(1,2))
        return U_mean   
        
    def compute_grid(self):
        U=[None]*len(self.path) 
        V=[None]*len(self.path) 
        for ti, pathi in enumerate (self.path):
            X, Y, dx, dy, U[ti], V[ti] = compute_grid(self.xs, self.ys, self.deltaxs[ti], self.deltays[ti])
        return X, Y, dx, dy, U, V  
        
    def compute_derivatives(self, edge_order=2):   
        dUdx = np.zeros(np.shape(self.U))
        dUdy = np.zeros(np.shape(self.U))
        dVdx = np.zeros(np.shape(self.U)) 
        dVdy = np.zeros(np.shape(self.U)) 
        for ti, pathi in enumerate (self.path):
            dUdx[ti], dUdy[ti], dVdx[ti], dVdy[ti] = compute_derivatives(self.dx, self.dy, self.U[ti], self.V[ti], edge_order=2)
        return dUdx, dUdy, dVdx, dVdy
                
    def compute_rot(self, edge_order=2):
        if ~hasattr(self, 'dUdx'):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(edge_order = edge_order)
        rot = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate (self.path):
            rot[ti] = compute_rot(self.dUdy[ti], self.dVdx[ti])
        return rot
    
    def compute_div(self, edge_order=2):
        if ~hasattr(self, 'dUdx'):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(edge_order=edge_order)   
        div = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate (self.path):
            div[ti] = compute_div(self.dUdx[ti], self.dVdy[ti])
        return div 
    
    def compute_ken(self):         
        ken = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate (self.path):
            ken[ti] = compute_ken(self.U[ti], self.V[ti])
        return ken 
        
    def compute_norm(self, U, V):        
        norm = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate (self.path):
            norm[ti] = compute_norm(U[ti], V[ti])
        return norm

    def compute_temporal_fft(self):
        if not hasattr(self, 't'):
            print("please define time before perform temporal Fourier transform")
        else:
            fftU, omega, psdU = oneD_fourier_transform(self.t, self.U, axis=0, parseval=False)
            fftV, omega, psdV = oneD_fourier_transform(self.t, self.V, axis=0, parseval=False)
            
            if not hasattr(self, 'fft'):
                self.fft = DataObject()
            if not hasattr(self.fft, 'time'):
                time=DataObject()
                self.fft.time = time
                
            self.fft.time.omega = omega
            self.fft.time.fftU = fftU        
            self.fft.time.fftV = fftV
            self.fft.time.psdU = psdU        
            self.fft.time.psdV = psdV
            
            
            
            
    def compute_spatial_fft(self):
        
        fftU, kx, ky, psdU = twoD_fourier_transform(self.X, self.Y, self.U, axis=(1,2), parseval=False)
        fftV, kx, ky, psdV = twoD_fourier_transform(self.X, self.Y, self.V, axis=(1,2), parseval=False)
        
        if not hasattr(self, 'fft'):
            self.fft = DataObject()
        if not hasattr(self.fft, 'spatial'):
            spatial=DataObject()
            self.fft.spatial = spatial        
        
        self.fft.spatial.kx = kx
        self.fft.spatial.ky = ky        
        self.fft.spatial.fftU = fftU        
        self.fft.spatial.fftV = fftV
        self.fft.spatial.psdU = psdU        
        self.fft.spatial.psdV = psdV   

    def compute_spatiotemp_fft(self):
        if hasattr(self, 'fft.spatial'):
            fftU, omega, psdU = oneD_fourier_transform(self.t, 
                                                       self.fft.spatial.fftU, 
                                                       axis=0, parseval=False)
            fftV, omega, psdV = oneD_fourier_transform(self.t, 
                                                       self.fft.spatial.fftV, 
                                                       axis=0, parseval=False)    
            kx = self.fft.spatial.kx
            ky = self.fft.spatial.ky
            
        elif hasattr(self, 'fft.time'):
            fftU, kx, ky, psdU = twoD_fourier_transform(self.X, self.Y, 
                                                        self.fft.time.fftU, 
                                                        axis=(1,2), parseval=False)
            fftV, kx, ky, psdV = twoD_fourier_transform(self.X, self.Y, 
                                                        self.fft.time.fftV, 
                                                        axis=(1,2), parseval=False)
            omega = self.fft.time.omega
        else:
            self.compute_temporal_fft()
            fftU, kx, ky, psdU = twoD_fourier_transform(self.X, self.Y, 
                                                        self.fft.time.fftU, 
                                                        axis=(1,2), parseval=False)
            fftV, kx, ky, psdV = twoD_fourier_transform(self.X, self.Y, 
                                                        self.fft.time.fftV, 
                                                        axis=(1,2), parseval=False)
            omega = self.fft.time.omega
                                                        
        if not hasattr(self, 'fft'):
            self.fft = DataObject()
        if not hasattr(self.fft, 'spatiotemp'):
            spatiotemp=DataObject()
            self.fft.spatiotemp = spatiotemp
        
        self.fft.spatiotemp.omega = omega
        self.fft.spatiotemp.kx = kx
        self.fft.spatiotemp.ky = ky
        self.fft.spatiotemp.fftU = fftU        
        self.fft.spatiotemp.fftV = fftV
        self.fft.spatiotemp.psdU = psdU        
        self.fft.spatiotemp.psdV = psdV
        
        