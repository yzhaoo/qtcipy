
### NB, this assumes a single orbital model

## if you have a multiorbital model, this will likely not work

import numpy as np
from .hubbard import get_density_i


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




def get_den_kpm_qtci(h,dim=1,**kwargs):
    """Return the full density using QTCI"""
    if dim==1: # 1d geometry
        return get_den_kpm_qtci_1d(h,**kwargs)
    elif dim==2: # 2d geometry
        return get_den_kpm_qtci_2d(h,**kwargs)



def get_den_kpm_qtci_1d(h,info_qtci=False,log=None,**kwargs):
    """Return the electronic density of the system uisng KPM and QTCI"""
    @memoize
    def f(i): # function to interpolate
        return get_density_i(h,i=int(i),**kwargs)
    xlim = [0,h.shape[0]] # limits of the interpolation
    nb = np.log(h.shape[0])/np.log(2) # number of sites
    if np.abs(int(nb)-nb)>1e-5:
        print("Number of points must be a power of 2")
        raise
    nb = int(nb) # number of points
    from .. import interpolate
    IP = interpolate.Interpolator(f,tol=1e-3,nb=nb,xlim=xlim,dim=1)
    if log is not None: # make a log
        rse,zse = IP.get_evaluated()
        log["QTCI_eval"].append(len(rse)/h.shape[0]) # ratio of evaluations
    if info_qtci:
         print(len(rse)/h.shape[0],"ratio of evaluations")
    return np.array([IP(float(i)) for i in range(0,h.shape[0])])







def get_den_kpm_qtci_2d(h,info_qtci=False,**kwargs):
    """Return the electronic density of the system uisng KPM and QTCI"""
    ncells = int(np.round(np.sqrt(h.shape[0]))) # number of cells
#    @memoize
    def f(ii,jj): # function to interpolate
        i = ncells*ii + jj # convert 2d index to 1d
        return get_density_i(h,i=int(i),**kwargs)
    xlim = [0,ncells] # limits of the interpolation
    ylim = [0,ncells] # limits of the interpolation
    nb = np.log(h.shape[0])/np.log(2)/2. # number of sites
    if np.abs(int(nb)-nb)>1e-5:
        print("Number of points must be a power of 2")
        raise
    nb = int(nb) # number of points
    from . import interpolate
    IP = interpolate.Interpolator(f,tol=1e-3,nb=nb,xlim=xlim,
            ylim=ylim,dim=2)
    if info_qtci:
         print(len(rse)/h.shape[0],"ratio of evaluations")
    out = np.zeros(h.shape[0]) # initialize
    for i in range(ncells*ncells): # loop over sites
        ii = i//ncells # floor division
        jj = i%ncells # residue
        out[i] = IP(float(ii),float(jj))
    return out # return the interpoalted result


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



