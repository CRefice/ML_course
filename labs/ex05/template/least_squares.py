# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    tx_t = tx.T
    a = tx_t.dot(tx)
    b = tx_t.dot(y)

    return np.linalg.solve(a, b)
