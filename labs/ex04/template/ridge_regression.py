# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N, D = tx.shape
    lambda_ *= 2 * N
    mat = tx.T @ tx + lambda_ * np.eye(D)
    w = np.linalg.solve(mat, tx.T @ y)
    return compute_mse(y, tx, w), w