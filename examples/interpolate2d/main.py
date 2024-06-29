import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy import interpolate

# Define a Python function to be passed to Julia
import numpy as np

def f(x,y):
#    return np.cos(30*x)/(1.+x**2)*(1+(x*y)**2)*np.sin(y)
    return 1./(1.+x**2+(3*y)**2)


xlim = [-4,4]
ylim = [-4,4]

# create the interpolator
IP = interpolate.Interpolator(f,tol=1e-2,nb=20,xlim=xlim,ylim=ylim,dim=2)



# check the evaluated points as a reference
rse,zse = IP.get_evaluated()
print(len(rse),"evaluations")
rse = np.array(rse)


# evaluate the function and the interpolation
nn = 100
xs = np.linspace(-3,3,nn)
ys = np.linspace(-3,3,nn)
xys = []
for x in xs:
    for y in ys:
        xys.append([x,y])

zs,zsi = [],[]
for xy in xys: 
    zs.append(f(xy[0],xy[1]))
    zsi.append(IP(xy[0],xy[1]))
#zsi = IP(xys)


print("Integral",IP.integrate())
#print("Integral",IP.integrate(axis=0))

import matplotlib.pyplot as plt
plt.subplot(1,3,1)
plt.imshow(np.array(zs).reshape(nn,nn).T)

plt.subplot(1,3,2)
plt.imshow(np.array(zsi).reshape(nn,nn).T)

plt.subplot(1,3,3)
plt.scatter(rse[:,0],rse[:,1],c=zse,s=10)
plt.axis("equal")

plt.tight_layout()


plt.show()



