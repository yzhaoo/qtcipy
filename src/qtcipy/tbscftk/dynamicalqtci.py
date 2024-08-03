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
    frac,qtci_kwargs = optimal_qtci(v,qtci_error=tol,kwargs0=kwargs)
    if qtci_kwargs is None: # none succeded
        print("No fitting QTCI found, using default")
        return get_default(v)
    else: 
#        print("Next iteration uses new QTCI found with fraction",frac)
#        print(qtci_kwargs)
        return qtci_kwargs

def overwrite_qtci_kwargs(kwargs,qtci_kwargs):
    for key in qtci_kwargs: # loop over all the keys
        kwargs[key] = qtci_kwargs[key] # overwrite
   #     print("Updating",key)




def get_default(v):
    """Return a default set of parameters for the QTCI"""
    qtci_kwargs = {"qtci_maxm":400} # reasonable guess
    qtci_kwargs["qtci_accumulative"] = True # accumulative mode
    qtci_kwargs["qtci_tol"] = 1e-2 # initial tol
    return qtci_kwargs # return this




from copy import deepcopy as cp


def initial_qtci_kwargs(SCF,**kwargs):
    """Return a reasonable initial guess for the kwargs
    of a QTCI for an SCF object"""
    if SCF.qtci_kwargs is None: # first iteration
        qtci_kwargs = {"qtci_maxm":400} # reasonable guess
        qtci_kwargs["qtci_accumulative"] = True # accumulative mode
        qtci_kwargs["qtci_tol"] = 1e-1 # initial tol
        SCF0 = SCF.copy() # make a copy
        SCF0.qtci_kwargs = qtci_kwargs # overwrite
        kw = cp(kwargs) # make a copy
        kw["kpm_delta"] = 2.0
        kw["delta"] = 2.0
        kw["use_qtci"] = True
        kw["use_kpm"] = True
        kw["backend"] = "C++"
        kw["maxite"] = 1 # one iteration
        SCF0.solve(**kw) # one iteration without accuracy
        print("SCF Initialization DONE")
        return SCF0.qtci_kwargs
    else: return SCF.qtci_kwargs # return this choice
















