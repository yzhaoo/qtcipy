
# class to perform SCF hubbard calculations
import numpy as np

class SCF_Hubbard():
    """Class to perform selfconsistency"""
    def __init__(self,H,MF=None,U=0.):
        """Initialize the SCF object"""
        self.H0 = H # store initial Hamiltonian
        self.H = [H.copy(),H.copy()] # store initial Hamiltonian
        if MF is None:
            dup0 = 0.2*np.array([(-1)**i for i in range(H.H.shape[0])])
            ddn0 = -dup0.copy() # initialize
            MF = [dup0,ddn0] # store
        self.MF = MF # store the mean-field guess
        set_Hubbard(self,U) # set the Hubbard
#        self.U = U # Hubbard interaction 
        self.Mz = MF[0]*0. # magnetization
        log = dict() # dictionary
        log["SCF_error"] = [] # initialize
        log["QTCI_eval"] = [] # initialize
        log["SCF_time"] = [] # initialize
        self.log = log # store the log
    def solve(self,**kwargs):
        """Perform the SCF loop"""
        from .hubbard import SCF_Hubbard
        h0 = self.H0.H
        U = self.U # Hubbard
        dup0 = self.MF[0].copy() # first mean-field
        ddn0 = self.MF[1].copy() # second mean-field
        hup1,hdn1,dup1,ddn1 = SCF_Hubbard(h0,ddn=ddn0,dup=dup0,U=U,
                log=self.log,**kwargs)
        self.H[0] = hup1.copy() # replace matrix
        self.H[1] = hdn1.copy() # replace matrix
        self.H = [hup1,hdn1] # store the resulting Hamiltonian
        self.MF = [dup1,ddn1] # store the densities
        self.Mz = dup1 - ddn1 # magnetization
        return self # return updated Hamiltonian
    def estimate_time(self,**kwargs):
        from .timeestimator import testimate
        return testimate(self.H0.H,**kwargs)
    def get_dos_i(self,w=None,**kwargs):
        """Compute the DOS in site i"""
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        (es,dup) = get_dos_i(self.H[0].H,x=w,**kwargs)
        (es,ddn) = get_dos_i(self.H[1].H,x=w,**kwargs)
        return es,dup+ddn # return energy and DOS
    def get_spin_dos_i(self,w=None,**kwargs):
        """Compute the DOS in site i"""
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        (es,dup) = get_dos_i(self.H[0].H,x=w,**kwargs)
        (es,ddn) = get_dos_i(self.H[1].H,x=w,**kwargs)
        return es,dup-ddn # return energy and DOS




def set_Hubbard(self,U):
    """Process the Hubbard interaction"""
    if type(U)==np.array: # if array
        if len(self.H.H)==U.shape[0]:
            self.U = U # store as array
        else: raise
    elif callable(U):
        Ua = [U(ri) for ri in self.H0.R] # callable
        self.U = Ua
    else: # assume that it is a float
        self.U = U # store as float



