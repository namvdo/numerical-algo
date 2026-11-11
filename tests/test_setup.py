"""Basic environment and package smoke tests."""

import importlib

import mpmath
import numba
import numpy as np
import scipy.linalg
import sympy


def test_src_module_import():
    mod = importlib.import_module("back_substitution")
    assert mod is not None


def test_numerical_dependencies():
    for name in ("numpy", "scipy", "numba", "mpmath", "sympy", "matplotlib"):
        assert importlib.import_module(name) is not None

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    Q, R = scipy.linalg.qr(A)
    assert np.allclose(Q @ R, A)

    @numba.njit
    def dot_manual(x, y):
        acc = 0.0
        for i in range(x.shape[0]):
            acc += x[i] * y[i]
        return acc

    assert abs(dot_manual(np.array([1.0, 2.0]), np.array([3.0, 4.0])) - 11.0) < 1e-12

    mpmath.mp.dps = 50
    assert mpmath.mpf(1) / 3 * 3 > mpmath.mpf("0.999999999999999999999999999999")
    assert set(sympy.Matrix([[1, 2], [2, 1]]).eigenvals().keys()) == {3, -1}
