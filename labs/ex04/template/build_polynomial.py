# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tx = x[:, np.newaxis]
    exponents = np.arange(degree)
    return np.power(tx, exponents)
