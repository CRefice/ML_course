# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    D = tx.shape[1]
    lambdap = 2*N*lambda_
    a = tx.T.dot(tx) + lambdap*np.eye(D)
    b = y.dot(tx)
    w_star = np.linalg.solve(a,b)
    return w_star
