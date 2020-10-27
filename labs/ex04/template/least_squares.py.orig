# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return compute_mse(y, tx, w), w
