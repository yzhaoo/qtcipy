import os
import sys

path = os.path.dirname(os.path.realpath(__file__))

# add xfacpy library
sys.path.append(path+"/pylib/xfac/build/python")

import xfacpy

class Interpolator():
    def __init__(self,f,xlim=[0.,1.],nb=20,tol=1e-2,**kwargs):
        """Initialize the interpolator object"""
        qgrid = xfacpy.QuanticsGrid(a=xlim[0],b=xlim[1], nBit=nb)  # build the quantics grid
        args = xfacpy.TensorCI2Param()                      # fix the max bond dimension
        args.bondDim = 15

        ci = xfacpy.QTensorCI(f1d=f, qgrid=qgrid, args=args)  # construct a tci of the quantics tensor
        while not ci.isDone():
            ci.iterate()
        self.ci = ci
        self.xlim = xlim
        self.qgrid = qgrid
#        self.ranks = ranks
#        self.dim = dim
#        self.tol = tol
        self.errors = ci.pivotError
        self.R = nb
        self.qtt = ci.get_qtt()  # the actual function approximating f
    def __call__(self,xs):
        out = [self.qtt.eval([x]) for x in xs]
        return out
    def integrate(self,axis=None,**kwargs):
        raise
    def get_evaluated(self):
        raise


