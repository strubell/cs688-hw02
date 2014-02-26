'''
CS688 HW02: Chain CRF

Question 4: Optimization warm-up

@author: Emma Strubell
'''

from scipy.optimize import minimize
import numpy as np

# objective function
def f(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

# gradient of objective function
def df(x): return np.array([400*x[0]*(x[1]-x[0]**2)+2*(1-x[0]), -200*(x[1]-x[0]**2)])

x0 = (0.0, 0.0)
result = minimize(f, x0, jac=df, tol=1e-10)
solution = result['x']
print "Value of max:", solution
print "Objective function evaluated at %s: %d" % (str(solution), f(solution))