import numpy as np

from ..qtcirecipes import optimal_qtci


def get_qtci_kwargs(kwargs,v,scf_error=None):
    """Overwrite the QTCI optional arguments according
    to how the mean field is evolving"""
    tol = 1e-2 # default tol
    if "qtci_tol" in kwargs: # overwrite if it is given
        tol = kwargs["qtci_tol"] # target tolerance
    else: # not given
        if scf_error is not None: # if given
            tol = min([tol,scf_error]) # overwrite
    # obtain the optimal QTCI for this data
#    frac,qtci_kwargs = optimal_qtci(v,qtci_error=tol)
    frac,qtci_kwargs = optimal_qtci(v,qtci_error=tol,kwargs0=kwargs)
    if qtci_kwargs is None: # none succeded
        print("No fitting QTCI found, using default")
        return {} # return the original ones
    else: 
#        print("Next iteration uses new QTCI found with fraction",frac)
#        print(qtci_kwargs)
        return qtci_kwargs

def overwrite_qtci_kwargs(kwargs,qtci_kwargs):
    for key in qtci_kwargs: # loop over all the keys
        kwargs[key] = qtci_kwargs[key] # overwrite
   #     print("Updating",key)







def initial_qtci_kwargs(SCF):
    """Return a reasonable initial guess for the kwargs
    of a QTCI for an SCF object"""
    qtci_kwargs = {"qtci_maxm":80} # reasonable guess
    qtci_kwargs["qtci_accumulative"] = True # acc mode
    qtci_kwargs["qtci_tol"] = 1e-2
    return qtci_kwargs # return this choice
















