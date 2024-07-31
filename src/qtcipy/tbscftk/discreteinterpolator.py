
import numpy as np

from ..interpolate import Interpolator as Interpolator_single

def Interpolator(f,norb=1,**kwargs):
    if norb==1: # one orbital
        return Interpolator_single(f,**kwargs) # conventional case
    else: # several orbitals
        return interpolate_norb(f,norb=norb,**kwargs) # conventional case




class Discrete_Interpolator():
    def __init__(self,IP):
        """Dummy class for a discrete interpolator, purely with Python
        objects"""
        self.frac = IP.frac
        self.nb = IP.nb
        x,y = IP.get_evaluated()
        self.x_ev = x.copy()
        self.y_ev = y.copy()
        self.error = IP.error
        self.opt_qtci_maxm = IP.opt_qtci_maxm
        self.out = [IP(i) for i in range(2**self.nb)]
    def __call__(self,i):
        return self.out[i]
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def get_evaluated(self):
        return self.x_ev,self.y_ev



def interpolate_norb(f,dim=1,norb=1,info_qtci=False,**kwargs):
    """Obtain an interpolator, where the function f has a certain
    number of orbitals"""
    IPs = [] # empty list
    ev = [] # evaluated points
    if info_qtci:
        print("#################################")
        print("### Multiorbital interpolator ###")
        print("#################################")
    def get_IP(iorb): # return the interpolator
        if dim==1: # one dimensional
            def fi(ii): return f(ii*norb + iorb) # redefine function
            IP = Interpolator_single(fi,dim=dim,
                    info_qtci = info_qtci,
                    **kwargs) # new interpolator
        else: raise # not implemented
        IP = Discrete_Interpolator(IP) # redefine
        return IP
    from .. import parallel
    IPs = parallel.pcall(get_IP,range(norb)) # call all
    IP = Interpolator_norb(IPs) # full interpolator
    return IP


class Interpolator_norb():
    def __init__(self,IPs): 
        # do nothing
        self.nb = IPs[0].nb # number of bits
        self.frac = np.mean([IP.frac for IP in IPs])
        self.opt_qtci_maxm = int(np.mean([IP.opt_qtci_maxm for IP in IPs]))
        self.error = np.mean([IP.error for IP in IPs])
        self.norb = len(IPs) # number of orbitals
        self.out = np.zeros(self.norb*(2**self.nb)) # initialize
        for iorb in range(self.norb): # loop
            for ii in range(2**self.nb): # loop over bits
                self.out[self.norb*ii + iorb] = IPs[iorb](ii) # call
        # store evaluated points
        nev = sum([len(IP.get_evaluated()[0]) for IP in IPs]) # number of evaluated points
        self.x_ev = np.zeros(nev) # initialize
        self.y_ev = np.zeros(nev) # initialize
        icount = 0
        for iorb in range(self.norb): # loop
            xs,ys = IPs[iorb].get_evaluated()
            for (x,y) in zip(xs,ys):
                self.x_ev[icount] = self.norb*x + iorb # store
                self.y_ev[icount] = y # store
    def __call__(self,i):
        return self.out[int(i)] # return result
    def get_evaluated(self):
        return self.x_ev,self.y_ev



