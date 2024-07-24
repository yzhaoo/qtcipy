
### NB, this assumes a single orbital model

## if you have a multiorbital model, this will likely not work

import numpy as np
from .hubbard import get_density_i

#
def memoize(func):
    """Decorator to use a cache"""
    cache = {}
    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func
#
#

from functools import lru_cache
#
#def memoize(func):
#    """Decorator to use a cache with LRU eviction policy"""
#    # Wrap the function with lru_cache
#    cached_func = lru_cache(maxsize=1000)(func)
#    return cached_func
#

#from functools import cache as memoize



def get_den_kpm_qtci(h,**kwargs):
    """Return the full density using QTCI"""
#        return get_den_kpm_qtci_1d(h,**kwargs)
    return get_den_kpm_qtci_general(h,**kwargs)
#    if dim==1: # 1d geometry
#        return get_den_kpm_qtci_1d(h,**kwargs)
#    elif dim==2: # 2d geometry
#        return get_den_kpm_qtci_2d(h,**kwargs)


#
#def get_den_kpm_qtci_1d(h,info_qtci=False,log=None,**kwargs):
#    """Return the electronic density of the system uisng KPM and QTCI"""
#    @memoize
#    def f(i): # function to interpolate
#        return get_density_i(h,i=int(i),**kwargs)
#    xlim = [0,h.shape[0]] # limits of the interpolation
#    nb = np.log(h.shape[0])/np.log(2) # number of pseudospin sites
#    nat = h.shape[0] # number of atoms
#    if np.abs(int(nb)-nb)>1e-5:
#        print("Number of points must be a power of 2")
#        raise
#    nb = int(nb) # number of points
#    from .. import interpolate
#    IP = interpolate.Interpolator(f,tol=1e-3,nb=nb,xlim=xlim,dim=1)
#    if log is not None: # make a log
#        rse,zse = IP.get_evaluated()
#        log["QTCI_eval"].append(len(rse)/h.shape[0]) # ratio of evaluations
#    if info_qtci:
#         print(len(rse)/h.shape[0],"ratio of evaluations")
#    out = np.zeros(nat,dtype=np.float_) # initialize
#    for i in range(nat): out[i] = IP(float(i)) # store result
#    return out # return the output
#
#



def get_den_kpm_qtci_general(h,info_qtci=False,log=None,**kwargs):
    """Return the electronic density of the system uisng KPM and QTCI"""
    f = get_function(h,**kwargs) # get the function to interpolate
    nb = get_nbits(h,**kwargs) # return the number of bits
    lim = get_lim(h,**kwargs) # get the limits
    if log is not None: qtci_maxm = log["opt_qtci_maxm"] # get the maxm
    else: qtci_maxm = 4 # reasonable guess
    IP = get_interpolator(h,f,nb,lim,
            qtci_maxm=qtci_maxm,
            **kwargs) # keyword arguments
    if log is not None: # make a log
        rse,zse = IP.get_evaluated()
        log["QTCI_eval"].append(len(rse)/h.shape[0]) # ratio of evaluations
#        log["opt_qtci_maxm"] = IP.opt_qtci_maxm # store
#    print(len(rse)/h.shape[0],"ratio of evaluations")
    if info_qtci:
         print(len(rse)/h.shape[0],"ratio of evaluations")
    out = evaluate_interpolator(h,IP,**kwargs) # evaluate the interpolator
    return out # return the output





def evaluate_interpolator(h,IP,dim=1,**kwargs):
    nat = h.shape[0]
    if dim==1: # 1D
        out = np.zeros(nat,dtype=np.float_) # initialize
        for i in range(nat): out[i] = IP(float(i)) # store result
        return out
    elif dim==2: # 2D
        print("Using 2D")
        n = int(np.sqrt(nat)) # lateral size of the system
        out = np.zeros(nat,dtype=np.float_) # initialize
        for i in range(n): 
            for j in range(n): 
                ii = n*i + j # index in real space
                out[ii] = IP(float(i),float(j)) # store result
        return out
    else: raise



# decorator to recover the Julia session
#from ..recover import retry
#from ..juliasession import restart as restart_julia

#@retry(initialize=restart_julia)
def get_interpolator(h,f,nb,lim,dim=1,qtci_tol=1e-3,**kwargs):
    """Return the interpolator"""
    from .. import interpolate
    if dim==1: # one dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim[0],
                dim=1,backend="C++",**kwargs)
    elif dim==2: # two dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim[0],
                ylim=lim[1],dim=2)
    else: raise # error otherwise
    return IP # return the interpolator




def get_lim(h,dim=1,**kwargs):
    """Return the limits"""
    if dim==1: # one dimensional
        xlim = [0,h.shape[0]] # limits of the interpolation
        return xlim,None
    elif dim==2: # two dimensional
        n = h.shape[0] # number of sites
        n = int(np.sqrt(n)) # lateral size of the system
        xlim = [0,n] # limits of the interpolation
        return xlim,xlim # return the limits
    else: raise # not implemented




def get_nbits(h,dim=1,**kwargs):
    """Get the number of required bits"""
    n = h.shape[0] # number of sites
    if dim==1: pass # ignore for 1d
    elif dim==2: # 2d
      n = int(np.sqrt(n)) # lateral size of the system
    else: raise
    nb = np.log(n)/np.log(2) # number of pseudospin sites
    if np.abs(int(nb)-nb)>1e-5:
        print("Number of points must be a power of 2")
        raise
    return int(nb) # return number of bits 



def get_function(h,dim=1,**kwargs):
    """Return the function to interpolate"""
#    @memoize
    def f1d(i): # function to interpolate
        ii = int(np.round(i)) # round the value
        if not 0<=ii<h.shape[0]: # fix and say
            print("WARNING, you are out of bounds!!!!")
            if ii<0: ii = 0 # fix
            else: ii = h.shape[0]-1 # last one
        return get_density_i(h,i=ii,**kwargs)
#    @memoize
    def f2d(i,j): # function to interpolate
        n = h.shape[0] # number of sites
        n = int(np.sqrt(n)) # lateral size of the system
        ii = n*i + j # index in real space
        return get_density_i(h,i=int(ii),**kwargs)
    if dim==1: return f1d
    elif dim==2: return f2d
    else: raise










import os ; import sys
sys.path.append(os.environ["PYQULAROOT"]) # pyqula

def get_dos_i(m,i=0,delta=1e-1,kpm_prec="single",
        kernel="jackson",npol_scale=2,**kwargs):
    """Return electronic density at site i"""
    ne = int(100/delta) # number of energies
    scale = 10.0 # scale of KPM method
#    delta = 0.1 # effective smearing
    from pyqula import kpm
    npol = int(npol_scale*scale/delta) # number of polynomials
    (es,ds) = kpm.ldos(m,i=i,ne=ne,kernel=kernel,kpm_prec=kpm_prec,
            npol=npol,**kwargs) # compute the LDOS with KPM
    return es,ds.real



