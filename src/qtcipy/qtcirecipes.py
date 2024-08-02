import numpy as np

from .tbscftk import discreteinterpolator

methods = ["maxm","accumulative"]
#methods = ["maxm"]


#### Acummulative mode has segfault randomly ####

def optimal_qtci(v,kwargs0=None,**kwargs):
    methods_f = [] # empty list
    if "maxm" in methods: methods_f.append(optimal_maxm) 
    if "accumulative" in methods: methods_f.append(optimal_accumulative) 
#    methods = [optimal_maxm] # these different methods
    outs = [] # storage
    if kwargs0 is not None: # if initial guess was given
#        try: # try the original parameters
            o = get_frac_args(v,**kwargs0)
            if o is not None: # this is ok
                frac0,kw0 = o[0],o[1]
                outs.append([frac0,kw0])
#            print("Previous QTCI",frac0)
#        except: pass # next one
    for method in methods_f:
#        try:
        out = method(v,**kwargs) # compute this one
#        except: out = None # if something wrong happened
        if out is not None: # sucess
            outs.append(out) # store
    if len(outs)>0: # if any succeded
        kwmin,fracmin = None,1.0
        for out in outs:
            if out[0]<fracmin:
                fracmin = out[0]
                kwmin = out[1]
#        print(kwmin)
        return fracmin,kwmin
    else: 
        return None,None # none succeded



def get_qtci_flags(kwargs):
    """Retain only the QTCI flags"""
    flags = ["qtci_maxm","qtci_accumulative","qtci_pivot1",
            "norb","qtci_tol","qtci_pivots","qtci_args"]
    out = dict()
    for key in kwargs: # loop
        if key in flags:
            out[key] = kwargs[key]
    return out




def get_frac_args(v,qtci_error=1e-2,**kwargs):
    """Get the fraction with a certain set of parameters"""
    nb = get_nbits(v,**kwargs)
    lim = get_lim(v,**kwargs)
    vi = v + qtci_error/100*(np.random.random()-0.5) # add noise
    f = lambda i: vi[int(i)] # function
    IP = get_interpolator(f,nb,lim,**kwargs) 
    erri = np.max(np.abs(np.array([IP(i) for i in range(len(v))]-v)))
    if erri<qtci_error: # this is ok, return 
        return IP.frac,IP.get_kwargs() # return the fraction
    else: return None   # this did not work




def get_interpolator(f,nb,lim,dim=1,backend="C++",
        qtci_tol=1e-2,**kwargs):
    """Return the interpolator"""
    from .tbscftk import discreteinterpolator as interpolate
    if dim==1: # one dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim,
                dim=1,backend=backend,**kwargs)
    elif dim==2: # two dimensional
        IP = interpolate.Interpolator(f,tol=qtci_tol,nb=nb,xlim=lim[0],
                ylim=lim[1],dim=2)
    else: raise # error otherwise
    return IP # return the interpolator




def get_nbits(v,norb=1,dim=1,**kwargs):
    """Get the number of required bits"""
    n = len(v) # number of sites
    if norb>1: n = n//norb # number of cells
    if dim==1: pass # ignore for 1d
    elif dim==2: # 2d
      n = int(np.sqrt(n)) # lateral size of the system
    else: raise
    nb = np.log(n)/np.log(2) # number of pseudospin sites
    if np.abs(int(nb)-nb)>1e-5:
        print("Number of points must be a power of 2")
        raise
    return int(nb) # return number of bits




def get_lim(v,dim=1,norb=1,**kwargs):
    """Return the limits"""
    if dim==1: # one dimensional
        n = len(v) # number of sites
        if norb>1: n = n//norb # by the number of orbitals
        xlim = [0,n] # limits of the interpolation
        return xlim
    elif dim==2: # two dimensional
        n = h.shape[0] # number of sites
        if norb>1: n = n//norb # by the number of orbitals
        n = int(np.sqrt(n)) # lateral size of the system
        xlim = [0,n] # limits of the interpolation
        return xlim,xlim # return the limits
    else: raise # not implemented











def optimal_maxm(v,qtci_error=1e-2):
    """Find the optimal QTCI with the non accumulative method"""
    ntries = 1 # average over these many tries
    maxmi = 3 # start with 3
    err,frac,pivot,maxm = 1e7,1.0,None,None # initialize
    if len(v)<10: norbs = [1]
    else: norbs = [1,2,4]
    weights = np.abs(v - np.mean(v)) + qtci_error # weight for pivots
    while True: # infinite loop over bond dimensions, until it works
        norbi = norbs[np.random.randint(len(norbs))] # one choice at random
#        for norbi in norbs: # loop over norbs
        nb = np.log(len(v)/norbi)/np.log(2) ; nb = int(nb) # integer value
        fullPivi = np.random.random()<0.5 # True or False randomly
        for it in range(ntries): # these many tries, pick the best
            #### generate global pivots ####
            use_gp = np.random.random()<0.5 # True or False randomly
            qtci_args = random_global_pivots(weights) # get global pivots
            ################################
            vi = v + qtci_error/100*(np.random.random(len(v)) - 0.5) # slightly random
            f = lambda i: vi[int(i)] # function to interpolate
            wi = [weights[norbi*i] for i in range(2**nb)] # redefine weight
            pi = pick_index(wi) # get this one 
            IP = discreteinterpolator.interpolate_norb(f,norb=norbi,xlim=[0,2**nb],
                                           nb=nb,backend="C++",
                                           qtci_tol = qtci_error,
                                           qtci_args = qtci_args,
                                           qtci_accumulative = False, # non acc mode
                                           qtci_fullPiv = fullPivi, # pivotting mode
                                           qtci_maxm=maxmi) # maxm
            erri = np.max(np.abs([IP(i) - f(i) for i in range(norbi*2**nb)])) # max error
            fraci = IP.frac # fraction of the space
            if erri<qtci_error: # desired error level has been reached
                if fraci<frac: # better fraction of space than the stored one
                    maxm = maxmi # store
                    norb = norbi # store
                    err = erri # store this error
                    frac = fraci # store this fraction
                    args = IP.qtci_args # any other arguments
                    pivot = IP.qtci_pivot1 # store this pivot
                    fullPiv = fullPivi
#                    print("Found new, maxm, norb, frac",maxm,norb,frac)
        maxmi = int(1.1*maxmi) + 1 # next bond dimension
        if maxmi>100: break # cutoff
    if pivot is None: return None # this did not work
    kwargs = dict() # dictionary with the options
    kwargs["qtci_maxm"] = maxm # store
    kwargs["qtci_accumulative"] = False # store
    kwargs["qtci_args"] = args # store
    kwargs["qtci_fullPiv"] = fullPiv # store
    kwargs["qtci_pivot1"] = pivot # store
    kwargs["norb"] = norb # store
#    print("Found optimal with fraction",frac,"error",err)
    return frac,kwargs # return optimal parameters




def random_global_pivots(v):
    """Generte random global pivots"""
    use_gp = np.random.random()<0.5 # True or False randomly
    qtci_args = {"qtci_use_global_pivots":use_gp} # store
    ntries = 5
    if use_gp: # use pivots
        n = len(v) # number of elements
        out = []
        for i in range(ntries): # one every 100
            out.append(pick_index(v)) # store
        out = np.unique(out) # get unique ones
        qtci_args["qtci_global_pivots_real"] = out
    return qtci_args # return




def optimal_accumulative(v,qtci_error=1e-2):
    """Find the optimal QTCI with the non accumulative method"""
    ntries = 1 # average over these many tries
    maxmi = 3 # start with 10
    err,frac,pivot,maxm = 1e7,1.0,None,None # initialize
    if len(v)<10: norbs = [1]
    else: norbs = [1,2,4]
    weights = np.abs(v - np.mean(v)) + qtci_error # weight for pivots
    while True: # infinite loop over bond dimensions, until it works
        norbi = norbs[np.random.randint(len(norbs))] # one choice at random
#        for norbi in norbs: # loop over norbs
        vi = v + qtci_error/10*(np.random.random(len(v)) - 0.5) # slightly random
        f = lambda i: vi[int(i)] # function to interpolate
        nb = np.log(len(v)/norbi)/np.log(2) ; nb = int(nb) # integer value
        for it in range(ntries): # these many tries, pick the best
            wi = [weights[norbi*i] for i in range(2**nb)] # redefine weight
            fullPivi = np.random.random()<0.5 # True or False randomly
            IP = discreteinterpolator.interpolate_norb(f,norb=norbi,xlim=[0,2**nb],
                                           nb=nb,backend="C++",
                                           qtci_tol = qtci_error, # tolerance
                                           qtci_accumulative = True, # non acc mode
                                           qtci_fullPiv = fullPivi, # pivotting mode
                                           qtci_maxm=maxmi) # maxm
            erri = np.max(np.abs([IP(i) - f(i) for i in range(norbi*2**nb)])) # max error
            fraci = IP.frac # fraction of the space
            if erri<qtci_error: # desired error level has been reached
                if fraci<frac: # better fraction of space than the stored one
                    maxm = maxmi # store
                    norb = norbi # store
                    pivot = IP.qtci_pivot1 # store this pivot
                    err = erri # store this error
                    frac = fraci # store this fraction
                    args = IP.qtci_args # store
                    fullPiv = fullPivi
#                    print("Found new, maxm, norb, frac",maxm,norb,frac)
        maxmi = int(1.1*maxmi) + 1 # next bond dimension
        if maxmi>100: break # cutoff
    if pivot is None: return None # this did not work
    kwargs = dict() # dictionary with the options
    kwargs["qtci_maxm"] = maxm # store
    kwargs["qtci_accumulative"] = True # store
    kwargs["qtci_pivot1"] = pivot # store
    kwargs["qtci_args"] = args # store
    kwargs["qtci_fullPiv"] = fullPiv # store
    kwargs["qtci_tol"] = qtci_error # store
    kwargs["norb"] = norb # store
#    print("Found optimal with fraction",frac,"error",err)
    return frac,kwargs # return optimal parameters







def pick_index(weights):
    """Return an index with probability weight[index]"""
    import random
    total = np.sum(weights)
    if total == 0:
        raise
    r = random.uniform(0, total)
    cumulative = 0
    for i, weight in enumerate(weights):
        cumulative += weight
        if r < cumulative:
            return i
    return len(weights) - 1  # In case of rounding issues

















