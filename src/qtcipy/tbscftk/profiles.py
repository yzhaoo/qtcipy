import numpy as np

# profiles for different SCF quantities


def envelop(H):
    """Envelop function"""
    R = H.R
    R[:,0] -= np.mean(R[:,0])
    R[:,1] -= np.mean(R[:,1])
    widthx = np.max(R[:,0])
    widthy = np.max(R[:,1])
    def envelop_i(i): # envelop for this site
        Wx = widthx*0.9
        Wy = widthy*0.9
        wallx = widthx/10 # width of the wall
        wally = widthy/10 # width of the wall
        fx = (-np.tanh((np.abs(R[i,0])-Wx)/wallx) + 1.0)/2.
        if np.max(H.R[:,1])>1e-2:
            fy = (-np.tanh((np.abs(R[i,1])-Wy)/wally) + 1.0)/2.
        else: fy = 1.0
        return fx*fy 
    return np.array([envelop_i(i) for i in range(len(R))]) # return



def get_profile(H,name):
    """Return the envelop of for a Hamiltonian given a name"""
    if name is None: return 1.0 # do nothing
    if name=="envelop":
        return envelop(H)




