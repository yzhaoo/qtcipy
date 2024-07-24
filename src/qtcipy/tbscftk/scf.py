
# class to perform SCF hubbard calculations
import numpy as np

class SCF_Hubbard():
    """Class to perform selfconsistency"""
    def __init__(self,H,MF=None,U=0.):
        """Initialize the SCF object"""
        self.H0 = H # store initial Hamiltonian
        self.H = [H.H.copy(),H.H.copy()] # store initial Hamiltonian
        if MF is None:
            dup0 = 0.2*H.AB # initialize
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
        log["opt_qtci_maxm"] = 5 # reasonable initial guess
        self.log = log # store the log
    def solve(self,qtci_maxm=None,**kwargs):
        """Perform the SCF loop"""
        if qtci_maxm is not None: # overwrite if needed (old used otherwise)
            self.log["opt_qtci_maxm"] = qtci_maxm
        from .hubbard import SCF_Hubbard
        h0 = self.H0.H
        U = self.U # Hubbard
        dup0 = self.MF[0].copy() # first mean-field
        ddn0 = self.MF[1].copy() # second mean-field
        hup1,hdn1,dup1,ddn1 = SCF_Hubbard(h0,ddn=ddn0,dup=dup0,U=U,
                dim=self.H0.dim, # dimensionality
                log=self.log,**kwargs)
        self.H[0] = hup1.copy() # replace matrix
        self.H[1] = hdn1.copy() # replace matrix
        self.H = [hup1,hdn1] # store the resulting Hamiltonian
        self.MF = [dup1,ddn1] # store the densities
        self.Mz = dup1 - ddn1 # magnetization
        return self # return updated Hamiltonian
    def estimate_time(self,**kwargs):
        from .timeestimator import testimate
        return testimate(self.H0.H,dim=self.H0.dim,**kwargs)
    def get_dos_i(self,w=None,**kwargs):
        """Compute the DOS in site i"""
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        (es,dup) = get_dos_i(self.H[0].H,x=w,**kwargs)
        (es,ddn) = get_dos_i(self.H[1].H,x=w,**kwargs)
        return es,dup+ddn # return energy and DOS
    def get_ldos(self,**kwargs):
        from .ldos import get_ldos
        oup = get_ldos(self.H[0],**kwargs)
        odn = get_ldos(self.H[1],**kwargs)
        return (oup+odn)/2. # up plus down
    def get_spin_ldos(self,**kwargs):
        from .ldos import get_ldos
        oup = get_ldos(self.H[0],**kwargs)
        odn = get_ldos(self.H[1],**kwargs)
        return (oup-odn)/2. # up minus down
    def get_spin_dos_i(self,w=None,**kwargs):
        """Compute the DOS in site i"""
        from .kpmrho import get_dos_i
        if w is None: w = np.linspace(-5.,5.,1000)
        (es,dup) = get_dos_i(self.H[0].H,x=w,**kwargs)
        (es,ddn) = get_dos_i(self.H[1].H,x=w,**kwargs)
        return es,dup-ddn # return energy and DOS
    def estimate_qtci_maxm(self,f,**kwargs):
        from .kpmrho import estimate_qtci_maxm
        return estimate_qtci_maxm(self.H0.H,self.H0.R,f,**kwargs)
    def save(self,**kwargs):
        from .saveload import save_SCF
        save_SCF(self,**kwargs)
    def load(self,**kwargs):
        from .saveload import load_SCF
        return load_SCF(**kwargs)




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



