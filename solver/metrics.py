import numpy as np

def absolute_error(exact, approx):
    return np.abs(exact - approx)

def max_error(exact, approx):
    return np.max(np.abs(exact - approx))

def mean_absolute_error(exact, approx):
    return np.mean(np.abs(exact - approx))

def strong_error(fine, coarse):
    return np.sqrt(np.mean((fine - coarse)**2))
