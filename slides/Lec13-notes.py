from scipy import optimize
import timeit

def f(x): 
  return np.exp(x[0]-1) + np.exp(-x[1]+1) + (x[0]-x[1])**2

def grad(x):
  return [
    np.exp(x[0]-1) + 2 * (x[0]-x[1]),
    -np.exp(-x[1]+1) - 2 * (x[0]-x[1])
  ]

def hess(x):
  return [
    [ np.exp(x[0]-1) + 2, -2                  ],
    [ -2                , np.exp(-x[1]+1) + 2 ]
  ]

x0 = [0, 0]

optimize.minimize(fun=f, x0=x0, jac=grad, method="BFGS")
optimize.minimize(fun=f, x0=x0, jac=grad, method="CG")
optimize.minimize(fun=f, x0=x0, jac=grad, method="Newton-CG")
optimize.minimize(fun=f, x0=x0, jac=grad, hess=hess, method="Newton-CG")
optimize.minimize(fun=f, x0=x0, jac=grad, method="Nelder-Mead")

timeit.Timer(lambda: optimize.minimize(fun=f, x0=x0, jac=grad, method="BFGS")).repeat(1, 100)
timeit.Timer(lambda: optimize.minimize(fun=f, x0=x0, jac=grad, method="CG")).repeat(1, 100)
timeit.Timer(lambda: optimize.minimize(fun=f, x0=x0, jac=grad, method="Newton-CG")).repeat(1, 100)
timeit.Timer(lambda: optimize.minimize(fun=f, x0=x0, jac=grad, hess=hess, method="Newton-CG")).repeat(1, 100)
timeit.Timer(lambda: optimize.minimize(fun=f, x0=x0, method="Nelder-Mead")).repeat(1, 100)
