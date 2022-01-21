## Setup

import numpy as np

rng = np.random.default_rng(1234)
n = 1000

## Create Data

X = np.hstack(
      [np.ones((n,1)), 
       rng.random((n,5))]
    )
beta = np.array([3.5, 1.8, -7.5, 0, 3, 9.8])
err = rng.normal(0, 0.1, size = n)

y = X @ beta + err

## Fit regression model - beta_hat =  (X^T X)^-1 X^Ty

np.linalg.inv(X.T @ X) @ X.T @ y

np.linalg.solve(X.T @ X, X.T @ y)

