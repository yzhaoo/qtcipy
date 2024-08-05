# function to compute the distance between two QTCI

import numpy as np


qtci_error = 1e-3 # default error in QTCI


def max_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.max(np.abs(a-b)) # maximum different


def mean_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.mean(np.abs(a-b)) # maximum different


default = "mean" # distance

def get_distance():
    if default =="max":
        return max_distance
    elif default =="mean":
        return mean_distance
    else: raise


