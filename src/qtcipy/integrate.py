from .juliasession import Main

def full_integrate(self):
    """Integrate in the full interval"""
    if self.dim==1:
        return Main.integrate_qtci_1d(self.ci,self.xlim[0],self.xlim[1],self.R)
    elif self.dim==2:
        return Main.integrate_qtci_2d(self.ci,self.xlim[0],self.xlim[1],
                  self.ylim[0],self.ylim[1],self.R)




def integrate_x(self,return_qtci=False):
    if self.dim!=2: raise # only for 2d
    from .interpolate import Interpolator
    def g(y): 
        """Define the new function you want to interpolate"""
        print("PTCI",y)
        def f(x): # this is the function to integrate
            return self(x,y) # return the value
        IPy = Interpolator(f,xlim=self.xlim,nb=self.R,tol=self.tol,
                dim=1)
        return IPy.integrate() # return the integral
    if return_qtci:
        IP = Interpolator(g,xlim=self.ylim,nb=self.R,tol=self.tol,
                    dim=1)
        return IP # return the interpolator
    else: 
        return g # return the bare function



def qtci_integrate(f,xlim=[0.,1.],tol=1e-6,nb=30,ylim=None):
    """Integrate a function using quantics tensor cross inteprolation"""
    from .interpolate import Interpolator
    if ylim is None:
        fci = Interpolator(f,tol=tol,nb=nb,xlim=xlim,dim=1)
    else:
        fci = Interpolator(f,tol=tol,nb=nb,xlim=xlim,
                ylim=ylim,dim=2)
    return fci.integrate()



