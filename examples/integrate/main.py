import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy import integrate

def f(x):
    delta = 1e-6
    return delta/(x**2 + delta**2)
from scipy.integrate import quad
#o = quad(f,-10,10)
o = integrate.qtci_integrate(f,xlim=[-100,100])
print(o)
