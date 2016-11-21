import numpy as np
def f(w):   
    f=6*w-11*np.log(1+np.exp(w))   
    return f
from sympy import *
def fp(w):   
    deriv =6-11*(np.exp(w)/(1+np.exp(w)))   
    return deriv
from scipy.optimize import minimize
mll = minimize(fun=lambda x: -f(x),x0=1,jac=lambda x: -fp(x),method='BFGS')
print(mll)
