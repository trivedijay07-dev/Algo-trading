"""Dependency-free Black-Scholes helpers for gamma / delta / IV work.

Vectorized over numpy arrays; normal pdf/cdf implemented via ``math.erf`` so
there is no scipy requirement.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_erf_vec = np.vectorize(math.erf)


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf_vec(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _d1(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_gamma(S, K, T, r, sigma) -> np.ndarray:
    """Black-Scholes gamma (same for calls and puts)."""
    T = np.maximum(np.asarray(T, dtype=float), 1e-9)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    d1 = _d1(S, K, T, r, sigma)
    return norm_pdf(d1) / (np.asarray(S, dtype=float) * sigma * np.sqrt(T))


def bs_delta(S, K, T, r, sigma, is_call: np.ndarray) -> np.ndarray:
    """Black-Scholes delta; ``is_call`` is a boolean array."""
    d1 = _d1(S, K, T, r, sigma)
    call_delta = norm_cdf(d1)
    return np.where(is_call, call_delta, call_delta - 1.0)
