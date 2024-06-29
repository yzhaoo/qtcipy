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




