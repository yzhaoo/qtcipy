import os
import sys

path = os.path.dirname(os.path.realpath(__file__))

# add xfacpy library
sys.path.append(path+"/pylib/xfac/build/python")

import xfacpy

class Interpolator():
    def __init__(self,f,xlim=[0.,1.],nb=20,
            **kwargs):
        """Initialize the interpolator object"""
        qgrid = xfacpy.QuanticsGrid(a=xlim[0],b=xlim[1], nBit=nb)  # build the quantics grid
        self.f = memoize(f)
        ci,args,opt_qtci_maxm = get_ci(self.f,nb=nb,qgrid=qgrid, **kwargs)
        self.ci = ci
        self.nb = nb
        self.xlim = xlim
        self.qgrid = qgrid
        self.qtci_maxm = args.bondDim # store the bond dimension
        self.errors = ci.pivotError
        self.error = err = ci.pivotError[len(ci.pivotError)-1]
        self.opt_qtci_maxm = opt_qtci_maxm
        self.R = nb
        rse,zse = self.get_evaluated()
        self.frac = len(rse)/(2**nb)
#        self.qtt = ci.get_qtt()  # the actual function approximating f
        self.qtt = ci.tt  # the actual function approximating f
    def __call__(self,xs):
        if is_iterable(xs):
            out = []
            for x in xs:
                xi = self.qgrid.coord_to_id([x])
                oi = self.qtt.eval(xi)
                out.append(oi)
        else:
            xi = self.qgrid.coord_to_id([xs])
            out = self.qtt.eval(xi)
        return out
    def integrate(self,axis=None,**kwargs):
        raise
    def get_evaluated(self):
        return get_cache_info(self.f)
    def get_eval_frac(self):
        return (len(self.get_evaluated()[0]))/(2**self.nb)



class CI():
    """Class for an enriched CI object"""
    def __init__(self,ci,qgrid=None,nb=1):
        self.xfac_ci = ci # original ci from xfac
        self.qgrid = qgrid # qgrid
        self.nb = nb # number of bits
    def iterate(self):
        """Make an iteration"""
        self.xfac_ci.iterate() 
    def __call__(self,x):
        return eval_ci(self.xfac_ci,self.grid)



def eval_ci(ci,qgrid,x):
    """Evaluate a CI"""
    xi = qgrid.coord_to_id([x]) # to grid
    return ci.tt.eval(xi) # evaluate this point







import numpy as np


def get_ci(f, qgrid=None, nb=1,
        qtci_recursive = False,
        qtci_pivot1 = None, # no initial guess
        qtci_maxm=20, # initial bond dimension
        info_qtci = False,
        qtci_tol = 1e-3, # tolerance of quantics
        qtci_accumulative = False,
        qtci_fullPiv = False,
        tol=None,**kwargs):
    """Compute the CI, using an iterative procedure if needed"""
    maxm = qtci_maxm # initialize
    pivots = [] # list of pivots
    if tol is not None:  qtci_tol = tol
    while True: # infinite loop until convergence is reached
        args = xfacpy.TensorCI2Param()  # fix the max bond dimension
        args.bondDim = maxm
        if qtci_pivot1 is None: args.pivot1 = qgrid.coord_to_id([0.]) # in the first point
        else: args.pivot1 = qgrid.coord_to_id([qtci_pivot1]) # in the first point
        args.fullPiv = qtci_fullPiv # search the full Pi matrix
        ci = xfacpy.QTensorCI(f1d=f, qgrid=qgrid, args=args)  # construct a tci
        if qtci_accumulative: # accumulative mode
            ci = accumulative_train(ci,qtci_tol=qtci_tol,nb=nb,f=f)
        ### reuse the previous pivots
#        x_used,y_used = get_cache_info(f) # return the evaluated points
#        x_used = [qgrid.coord_to_id([x]) for x in x_used] # transform to quantics
#        print(pivots)
#        pivots = []
#        if len(pivots)!=0: # pivots provided
#            for ii in range(len(pivots)): # loop over bits
#                ci.addPivotsAt(pivots[ii],ii) # add the previous pivots
#        ci.addPivotsAllBonds(x_used) # add the global pivots
#        print(x_used) ; exit()
#        if len(x_used)>0:
#            ci.addPivotPoints(x_used) # add the global pivots
#            ci.addPivotValues(y_used) # add the global pivots
        # train quantics #
        else: # conventional mode
            while not ci.isDone(): # iterate until convergence
                ci.iterate()
                err = ci.pivotError[len(ci.pivotError)-1]
                if tol is not None: # if tol given, break when tol reached
                    if err<tol: # tolerance reached
                        if info_qtci:
                            print("QTCI pivot tol reached",err," stopping training")
                        break # stop loop
        # evaluate error #
        evf = len(get_cache_info(f)[0])/(2**nb) # percentage of evaluations
        err = ci.pivotError[len(ci.pivotError)-1] # error
        if info_qtci:
            print("Eval frac = ",evf,"maxm = ",maxm,"error = ",err,"target = ",tol)
        if tol is not None: # if enforce an error, check for convergence
            if err<tol: break # if convergence, break
            else: # no convergence
                if not qtci_recursive: break # no recursive, just stop
                else: # recursive mode
                    maxm = maxm + 1 # for next iteration
                    print("Recursive QTCI, next bond dim",maxm)
        else: break # no error enforced, just stop
        ### for the next iteration ###
#        ci.getPivotsAt(0)
#        pivots = [ci.getPivotsAt(ii) for ii in range(nb-2)] # get all the pivots used before
#        print(pivots) ; exit()
    return ci,args,maxm # return the optimal bond dimension, for next iteration




def accumulative_train(ci,qtci_tol=1e-3,qgrid=None,
        nb=1,f=None):
#    print("Accumulative mode")
    ci = xfacpy.to_tci1(ci) # to type one
    batch = 3 # three times
    while True: # infinite loop
        tols = []
        for j in range(batch): # do this once
            ci.iterate(1) # iterate
            toli = ci.pivotError[len(ci.pivotError)-1] # tolerance in evaluated points
            tols.append(toli) # store
        if np.mean(tols)<qtci_tol: 
#            print("Accumulative reached error, stopping",np.min(tols))
            break # stop loop
        # other stopping criteria
        npoints = 2**nb
        nev = len(get_cache_info(f)[0])
        if nev/npoints>0.95: 
#            print("Almost all points evaluated, stopping")
            break
    ci = xfacpy.to_tci2(ci) # to type two
    return ci # return







from collections.abc import Iterable
def is_iterable(e): return isinstance(e,Iterable)



import pickle
from functools import lru_cache, wraps

def memoize(f):
    @lru_cache(maxsize=None)
    @wraps(f)
    def memoized_func(*args, **kwargs):
        return f(*args, **kwargs)

    memoized_func._cache = {}

    original_func = memoized_func.__wrapped__

    @wraps(memoized_func)
    def wrapper(*args, **kwargs):
        result = memoized_func(*args, **kwargs)
        key = pickle.dumps((args, frozenset(kwargs.items())))
        memoized_func._cache[key] = result
        return result

    wrapper.cache_info = memoized_func.cache_info
    wrapper._cache = memoized_func._cache

    return wrapper

# Define the function to be memoized
def f(x):
    return x**2

# Memoize the function
#fo = memoize(f)

# Recover the list of evaluated points
def get_cache_info(func):
    cache_info = func.cache_info()
    cache_keys = list(func._cache.keys())
    xs = [pickle.loads(key) for key in cache_keys]
    xs = [x[0][0] for x in xs]
    ys = [func(x) for x in xs]
    return xs,ys






