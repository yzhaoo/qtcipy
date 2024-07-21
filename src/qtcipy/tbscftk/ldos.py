

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




