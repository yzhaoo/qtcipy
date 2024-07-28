import os
from .juliasession import Main

class Interpolator():
    def __init__(self,f,xlim=[0.,1.],nb=20,tol=1e-2,dim=1,
            ylim=[0.,1.],zlim=[0.,1.],guess=None,**kwargs):
        """Initialize the interpolator object"""
# Pass the Python function and variables to the Julia function
        if dim==1:
            ci,ranks,errors,qgrid = Main.initialize_interpolator_1d(f, xlim[0], xlim[1],nb,tol)
        elif dim==2:
            ci,ranks,errors,qgrid = Main.initialize_interpolator_2d(f, 
                    xlim[0], xlim[1],
                    ylim[0], ylim[1],
                    nb,tol)
        elif dim==3:
            ci,ranks,errors,qgrid = Main.initialize_interpolator_3d(f, 
                    xlim[0], xlim[1],
                    ylim[0], ylim[1],
                    zlim[0], zlim[1],
                    nb,tol)
        self.ci = ci
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.ranks = ranks
        self.opt_qtci_maxm = ranks[-1]
        self.dim = dim
        self.tol = tol
        self.errors = errors
        self.error = errors[-1]
        self.qgrid = qgrid
        self.R = nb
    def __call__(self,xs,y=None,z=None):
        if self.dim==1:
            ys = Main.call_qtci_1d(xs, self.qgrid, self.ci)
        elif self.dim==2:
            ys = Main.call_qtci_2d(xs,y, self.qgrid, self.ci)
        elif self.dim==3:
            ys = Main.call_qtci_3d(xs,y,z, self.qgrid, self.ci)
        else: raise
        return ys
    def integrate(self,axis=None,**kwargs):
        if axis is None:
            from .integrate import full_integrate
            return full_integrate(self)
        elif axis==0:
            from .integrate import integrate_x
            return integrate_x(self,**kwargs)
        else: raise
    def get_evaluated(self):
        return Main.evaluated_points(self.ci)


