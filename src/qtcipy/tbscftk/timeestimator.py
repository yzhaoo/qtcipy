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
#    out = np.array([IP(float(i)) for i in range(0,h.shape[0])])
    t1 = time.time() # get the time
    rse,zse = IP.get_evaluated()
    fac = len(rse)/h.shape[0] # ratio of evaluations
    return fac,t1-t0 # return factor of evaluations, and time



def get_function_dummy(h,dim=1,**kwargs):
    """Return the function to interpolate"""
    def f1d(i): # function to interpolate
        i = int(i)
        return h[i,i] # element of the matrix
    def f2d(i,j): # function to interpolate
        n = h.shape[0] # number of sites
        n = int(np.sqrt(n)) # lateral size of the system
        ii = n*i + j # index in real space
        return h[ii,ii] # element of the matrix
    if dim==1: return f1d
    elif dim==2: return f2d
    else: raise



def testimate_qtci_general(h,dim=1,**kwargs):
    """Return the electronic density of the system uisng KPM and QTCI"""
    f = get_function_dummy(h,**kwargs) # get the function to interpolate
    from .kpmrho import get_nbits,get_lim,get_interpolator
    nb = get_nbits(h,**kwargs) # return the number of bits
    lim = get_lim(h,**kwargs) # get the limits
    t0 = time.time() # initial time
    IP = get_interpolator(h,f,nb,lim,**kwargs) # keyword arguments
    t1 = time.time() # get the time
    rse,zse = IP.get_evaluated()
    fac = len(rse)/h.shape[0] # ratio of evaluations
    return fac,t1-t0 # return factor of evaluations, and time



def testimate(h,estimate_rho=True,**kwargs):
    """Estimate the total time of a 1D QTCI"""
    fac,dt = testimate_qtci_general(h,**kwargs) # get factor of evaluated and time
    from .hubbard import get_density_i
    if estimate_rho:
        t0 = time.time() # initial time
        get_density_i(h,**kwargs)
        t1 = time.time() # initial time
        T = (t1-t0)*fac*h.shape[0] # expected total time
    else: T = dt # time is just the quantics time 
    return T,fac # return expected time and factor of evaluations




