"""Compute layer: numerics that generate figure geometry.

Everything returns plain arrays in math coordinates. No drawing.
"""

from __future__ import annotations

import numpy as np


def exp_partial_sums(z: complex, n_terms: int) -> np.ndarray:
    """Cumulative partial sums S_0=0, S_{k+1}=S_k + z^k/k! as (n_terms+1, 2) points.

    The polyline S_0 -> S_1 -> ... is the vector-chain picture of the
    exponential series (Needham Fig [9]).
    """
    k = np.arange(n_terms)
    terms = np.empty(n_terms, dtype=complex)
    terms[0] = 1.0
    if n_terms > 1:
        terms[1:] = np.cumprod(z / k[1:])
    sums = np.concatenate([[0.0 + 0j], np.cumsum(terms)])
    return np.column_stack([sums.real, sums.imag])
