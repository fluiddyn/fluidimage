# TO DO
#####
# ADD epsilon, reynolds stresses etc...


import numpy as np

def compute_grid(xs, ys, deltaxs, deltays):
                
    x = np.unique(xs)
    y = np.unique(ys)
    X, Y = np.meshgrid(x,y)
    X = X.transpose()
    Y = Y.transpose()
    U=np.reshape(deltaxs,(x.size,y.size))
    V=np.reshape(deltays,(x.size,y.size))       
        
    X = X
    Y = Y
    dx = X[1][0]-X[0][0]
    dy = Y[0][1]-Y[0][0]
    U = U
    V = V
    return X, Y, dx, dy, U, V

def compute_derivatives(dx, dy, U, V, edge_order=2):
        
    dUdx= np.gradient(U, dx, edge_order=edge_order)[0]
    dUdy= np.gradient(U, dy, edge_order=edge_order)[1]
    dVdx = np.gradient(V, dx, edge_order=edge_order)[0]    
    dVdy = np.gradient(V, dy, edge_order=edge_order)[1]

    return dUdx, dUdy, dVdx, dVdy
                
def compute_rot(dUdy, dVdx): 
    rot = dVdx - dUdy
    return rot
     
def compute_div(dUdx, dVdy):
    div = dUdx + dVdy  
    return div
        
def compute_ken(U, V):           
    ken = (U**2 + V**2 )/2 
    return ken
    
def compute_norm(U, V):           
    norm = np.sqrt(U**2 + V**2) 
    return norm