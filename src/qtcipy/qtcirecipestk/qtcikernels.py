
# kernel functions to redefine the function to be interpolated

import numpy as np

def random_kernel():
    """Return a random QTCI kernel"""
    return {"qtci_power_kernel": 0.3 + 1.7*np.random.random()}



def power_kernel(p,inverse=False):
    """Redefine a function through a power"""
    if inverse: p = 1./p # inverse power
    def f(x):
        s = x/(np.abs(x) + 1e-6)
        return s*np.power(np.abs(x),p) # power
    return f # return function



def get_kernel(kw,**kwargs):
    if "qtci_power_kernel" in kw: # using the power kernel
        p = kw["qtci_power_kernel"] # get the power
        return power_kernel(p,**kwargs) # return this kernel
    else:
        return lambda x: x # do nothing





