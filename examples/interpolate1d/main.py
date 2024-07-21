import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy import interpolate

# Define a Python function to be passed to Julia
import numpy as np

def f(x):
    return np.cos(30*x)/(1.+x**2)


xlim = [-4,4]
#from scipy import integrate
#out = integrate.quad(f,xlim[0],xlim[1])
#print("Integral with quad, as a reference",out[0])

# create the interpolator
IP = interpolate.Interpolator(f,tol=1e-3,nb=10,xlim=xlim)

# evaluate the function and the interpolation
xs = np.linspace(-3,3,400)
ys = f(xs)
ysi = IP(xs)

# check the evaluated points as a reference
#xse,yse = IP.get_evaluated()
#print("Integral",IP.integrate())


#print(len(xse),"evaluations")

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.plot(xs,ys,label="Original")
#plt.scatter(xs,ysi,label="interpolated")
plt.legend()

plt.subplot(1,2,2)
plt.plot(xs,ys,label="Original")
#plt.scatter(xse,yse,label="evaluated")
plt.legend()

plt.show()


