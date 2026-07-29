"""Compute layer: numerics that generate figure geometry.

Everything returns plain arrays in math coordinates. No drawing.
"""

from __future__ import annotations

import numpy as np


def cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z-component of the 2D cross product, broadcast over (..., 2) arrays.

    numpy 2.x removed np.cross for 2-vectors; this is the replacement for
    every inline `ax*by - ay*bx` site."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_in_poly(pt: tuple[float, float], poly: np.ndarray) -> bool:
    """Even-odd ray cast: is pt inside the closed polygon (N, 2)?"""
    x, y = float(pt[0]), float(pt[1])
    p = np.asarray(poly, dtype=float)
    x0, y0 = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
    return bool(np.count_nonzero(crosses & (x < xi)) % 2)


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


# ---------------------------------------------------------------------------
# Diffusion: OU forward process on a 1D Gaussian mixture — everything analytic.
# Forward SDE dx = -x dt + sqrt(2) dW, kernel x_t|x_0 ~ N(x0 e^{-t}, 1-e^{-2t}).


from dataclasses import dataclass


@dataclass(frozen=True)
class Mixture1D:
    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray


def ou_marginal(m0: Mixture1D, t: float) -> Mixture1D:
    decay = np.exp(-t)
    var = m0.sigmas**2 * decay**2 + (1.0 - decay**2)
    return Mixture1D(m0.weights, m0.means * decay, np.sqrt(var))


def mixture_pdf(m: Mixture1D, x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)[:, None]
    comp = np.exp(-0.5 * ((x - m.means) / m.sigmas) ** 2) / (m.sigmas * np.sqrt(2 * np.pi))
    return (comp * m.weights).sum(axis=1)


def mixture_score(m: Mixture1D, x: np.ndarray) -> np.ndarray:
    """d/dx log p(x) for a Gaussian mixture."""
    x = np.atleast_1d(x)[:, None]
    comp = np.exp(-0.5 * ((x - m.means) / m.sigmas) ** 2) / (m.sigmas * np.sqrt(2 * np.pi))
    w = comp * m.weights
    num = (w * (m.means - x) / m.sigmas**2).sum(axis=1)
    return num / w.sum(axis=1)


def _pf_field(m0: Mixture1D, t: float, x: np.ndarray) -> np.ndarray:
    """Probability-flow ODE velocity dx/dt = -x - score_t(x)."""
    return -x - mixture_score(ou_marginal(m0, t), x)


def pf_ode_paths(m0: Mixture1D, x_T: np.ndarray, T: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """RK4-integrate the probability-flow ODE backward from t=T to 0.

    Returns (ts, paths) with ts (n_steps+1,) decreasing and paths
    (n_paths, n_steps+1)."""
    h = -T / n_steps
    ts = T + h * np.arange(n_steps + 1)
    paths = np.empty((len(x_T), n_steps + 1))
    paths[:, 0] = x_T
    x = np.asarray(x_T, dtype=float)
    for i in range(n_steps):
        t = ts[i]
        k1 = _pf_field(m0, t, x)
        k2 = _pf_field(m0, t + h / 2, x + h / 2 * k1)
        k3 = _pf_field(m0, t + h / 2, x + h / 2 * k2)
        k4 = _pf_field(m0, t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        paths[:, i + 1] = x
    return ts, paths


def reverse_sde_paths(m0: Mixture1D, x_T: np.ndarray, T: float, n_steps: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Euler–Maruyama on the reverse-time SDE dx = [-x - 2 score] dt + sqrt(2) dW̄."""
    h = T / n_steps
    ts = T - h * np.arange(n_steps + 1)
    paths = np.empty((len(x_T), n_steps + 1))
    paths[:, 0] = x_T
    x = np.asarray(x_T, dtype=float)
    for i in range(n_steps):
        t = ts[i]
        drift = -x - 2.0 * mixture_score(ou_marginal(m0, t), x)
        x = x - h * drift + np.sqrt(2 * h) * rng.standard_normal(len(x))
        paths[:, i + 1] = x
    return ts, paths


def sample_mixture(m: Mixture1D, n: int, rng: np.random.Generator) -> np.ndarray:
    comp = rng.choice(len(m.weights), size=n, p=m.weights)
    return m.means[comp] + m.sigmas[comp] * rng.standard_normal(n)


def cassinian_curves(k: float, n_samples: int = 400) -> list[np.ndarray]:
    """Cassinian ovals |z-1||z+1| = k^2 (foci at +-1), as closed polylines.

    Polar form about the origin: r^2 = cos(2t) +- sqrt(k^4 - sin^2(2t)).
    k < 1: two loops around the foci; k = 1: lemniscate; k > 1: one loop.
    """
    if k < 1.0:
        alpha = 0.5 * np.arcsin(k * k)
        t = np.linspace(-alpha + 1e-9, alpha - 1e-9, n_samples // 2)
        disc = np.sqrt(np.maximum(k**4 - np.sin(2 * t) ** 2, 0.0))
        r_out = np.sqrt(np.cos(2 * t) + disc)
        r_in = np.sqrt(np.maximum(np.cos(2 * t) - disc, 0.0))
        loop = np.concatenate([
            r_out[:, None] * np.column_stack([np.cos(t), np.sin(t)]),
            (r_in[::-1])[:, None] * np.column_stack([np.cos(t[::-1]), np.sin(t[::-1])]),
        ])
        return [loop, -loop]
    if abs(k - 1.0) < 1e-12:
        t = np.linspace(-np.pi / 4, np.pi / 4, n_samples)
        r = np.sqrt(np.maximum(2 * np.cos(2 * t), 0.0))
        loop = r[:, None] * np.column_stack([np.cos(t), np.sin(t)])
        return [loop, -loop]
    t = np.linspace(0, 2 * np.pi, n_samples)
    r = np.sqrt(np.cos(2 * t) + np.sqrt(k**4 - np.sin(2 * t) ** 2))
    return [r[:, None] * np.column_stack([np.cos(t), np.sin(t)])]
