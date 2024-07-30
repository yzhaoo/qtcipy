

def Interpolator(f,backend="Julia",**kwargs):
    """Wrapper for Julia and C++ TCI interpolators"""
    if backend=="C++":
        from .interpolatecpp import Interpolator as InterpolatorCpp
        return InterpolatorCpp(f,**kwargs)
    elif backend=="Julia":
        from .interpolatejulia import Interpolator as Interpolatorjl
        return Interpolatorjl(f,**kwargs)




