import os   
import numpy as np

from fluidimage.data_objects.piv import (MultipassPIVResults)

    
class PIV_Postproc(MultipassPIVResults):

    def __init__(self, path):
        super(PIV_Postproc, self).__init__(path)
        self.path = os.path.abspath(path)
        self.nb_passes = np.size(self.passes)


    
    # the grid is not regular...
    # need to save interpolated data    
    
    def compute_grid(self, passe):
                
        x = np.unique(self.passes[passe].xs)
        y = np.unique(self.passes[passe].ys)
        X, Y = np.meshgrid(self.passes[passe].xs,self.passes[passe].ys)
        U, V = np.meshgrid(self.passes[passe].deltaxs,self.passes[passe].deltays)
        return X, Y, U, V
    
    def compute_rot(self, passe=None):
        if passe == None:
            passe = self.nb_passes - 1
            
        X, Y, U, V = self.compute_grid(passe)
        gradVX = np.gradient(V, self.passes[passe].xs, edge_order=2)
        self.passes[passe].rot = gradVX.f
        return gradVX
        