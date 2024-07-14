import numpy as np
import time

def testimate1d_qtci(h):
    """Return the electronic density of the system uisng KPM and QTCI"""
    def f(i): # function to interpolate
        i = int(i)
        return h[i,i] # element of the matrix
    xlim = [0,h.shape[0]] # limits of the interpolation
    nb = np.log(h.shape[0])/np.log(2) # number of sites
    if np.abs(int(nb)-nb)>1e-5:
        print("Number of points must be a power of 2")
        raise
    nb = int(nb) # number of points
    from .. import interpolate
    t0 = time.time() # initial time
    IP = interpolate.Interpolator(f,tol=1e-3,nb=nb,xlim=xlim,dim=1)
    out = np.array([IP(float(i)) for i in range(0,h.shape[0])])
    t1 = time.time() # get the time
    rse,zse = IP.get_evaluated()
    fac = len(rse)/h.shape[0] # ratio of evaluations
    return fac,t1-t0 # return factor of evaluations, and time



def testimate(h,dim=1,norb=1,estimate_rho=True,**kwargs):
    """Estimate the total time of a 1D QTCI"""
    if norb !=1: raise
    if dim !=1: raise
    fac,dt = testimate1d_qtci(h) # get factor of evaluated and time
    from .hubbard import get_density_i
    if estimate_rho:
        t0 = time.time() # initial time
        get_density_i(h,**kwargs)
        t1 = time.time() # initial time
        T = (t1-t0)*fac*h.shape[0] # expected total time
    else: T = dt # time is just the quantics time 
    return T,fac # return expected time and factor of evaluations




