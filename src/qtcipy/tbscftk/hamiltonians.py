import numpy as np

class Hamiltonian():
    """Class for a Hamiltonian"""
    def __init__(self,dim=1,H=None,R=None):
        """Class for a Hamiltonian"""
        self.dim = 1 # dimensionality
        self.H = H # matrix
        self.R = R # locations
    def get_SCF_Hubbard(self,**kwargs):
        """Return the SCF object"""
        from .scf import SCF_Hubbard
        return SCF_Hubbard(self,**kwargs)
    def modify_hopping(self,*args,**kwargs):
        return modify_hopping(self,*args)
    def add_onsite(self,*args,**kwargs):
        return add_onsite(self,*args)
    def get_density_i(self,**kwargs):
        from .hubbard import get_density_i
        return get_density_i(self.H,**kwargs)
    def get_dos_i(self,w=None,**kwargs):
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        return get_dos_i(self.H,x=w,**kwargs)
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
#        return get_density_i(self.H,**kwargs)


def chain(L):
    """Hamiltonian of a chain"""
    # define a first neighbor tight binding model
    n = 2**L # number of sites
    rows,cols = np.array(np.arange(0,n-1)),np.array(np.arange(1,n)) # indexes
    data = np.zeros(n-1,dtype=np.complex_) # hopping 1 for all
    data[:] = 1.0 # initialize
    from scipy.sparse import csc_matrix
    h0 = csc_matrix((data,(rows,cols)),shape=(n,n),dtype=np.complex_) # create single particle hopping
    h0 = h0 + h0.T # add the transpose
    r = np.zeros((n,2),dtype=np.float_) # initialize
    r[:,0] = np.arange(n) # locations
    H = Hamiltonian(dim=1,H=h0,R=r) # create Hamiltonian
    return H # return the Hamiltonian




def modify_hopping(self,f,**kwargs):
    """Modify the hoppings of a Hamiltonian"""
    from scipy.sparse import coo_matrix
    m = self.H.copy() # get the matrix
    R = self.R # positions
    mo = coo_matrix(m) # to coo_matrix
    for i in range(len(mo.data)): # loop over non-zero elements
        ii = mo.row[i] # get this row
        jj = mo.col[i] # get this column
        d = mo.data[i] # get this data
        r = (R[ii] + R[jj])/2. # average location
        d1 = d + f(r) # add a contribution
        m[ii,jj] = d1 # store the new hopping
    self.H = m # store the new Hamiltonian



def add_onsite(self,f):
    """Add an onsite energy tot he Hamiltonian"""
    n = self.H.shape[0]
    rows,cols = np.array(range(n)),np.array(range(n)) # indexes
    data = np.array([f(ri) for ri in self.R],dtype=np.complex_) 
    from scipy.sparse import csc_matrix
    h0 = csc_matrix((data,(rows,cols)),shape=(n,n),dtype=np.complex_)
    self.H = self.H + h0 # add the onsite energy


