import numpy as np

import os ; import sys
sys.path.append(os.environ["PYQULAROOT"]) # pyqula


from pyqula.ldostk.sparseldos import get_ldos as get_ldos_sparse


def get_ldos(m,info_qtci=True,i=None,**kwargs):
    """Return the LDOS using sparse inversion"""
    if i is None: # compute all sites using quantics
        def f(ii): 
            o = get_ldos_sparse(m,i=int(ii),**kwargs)
            o = float(o)
            return o
#        return [f(ii) for ii in range(m.shape[0])]
        from .kpmrho import get_nbits,get_lim,get_interpolator
        from .kpmrho import evaluate_interpolator
        nb = get_nbits(m,**kwargs) # return the number of bits
        lim = get_lim(m,**kwargs) # get the limits
        IP = get_interpolator(m,f,nb,lim,**kwargs)
        out = evaluate_interpolator(m,IP,**kwargs) # evaluate the interpolator
        ## number of evaluations
        if info_qtci:
            rse,zse = IP.get_evaluated()
            fac = len(rse)/m.shape[0] # ratio of evaluations
            print("Ratio of evaluations in LDOS",fac)
        return np.array(out) # return the output
    else:
        return get_ldos_sparse(m,i=i,**kwargs) # return a single one


def get_dos_i(H,w=None,**kwargs):
    """DOS in one site"""
    from .kpmrho import get_dos_i
    if w is None: w = np.linspace(-5.,5.,1000)
    return get_dos_i(H,x=w,**kwargs)



def get_dos(m,i=None,ntries=10,**kwargs):
    """Return the DOS, averaging over several vectors"""
    o = []
    d0 = 0.
    if i is None: i = [ii for ii in range(m.shape[0])]
    for j in range(ntries):
        e,d = get_dos_i(m,i=i,**kwargs)
        d0 = d0 + d
    return e,d0/ntries



