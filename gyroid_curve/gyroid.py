"""The gyroid field -- the single source of truth for the whole pipeline.

Standard nodal (trigonometric) approximation to the Schoen G surface (spec 1.1):

    phi(x, y, z) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)

The surface is the level set ``phi == 0``.  It is 2*pi-periodic per axis; one
unit cell is ``[0, 2*pi)^3``.

CAVEAT (encode everywhere): this nodal surface is *not* the true minimal gyroid.
Its mean curvature oscillates about zero.  This is fine for generative art -- we
never claim minimality (spec 1.1, 7).

Every projection, normal and tangent frame in the pipeline is derived from
``phi`` and ``grad_phi`` here, so there is exactly one definition of the surface.
"""

from __future__ import annotations

import numpy as np


def phi(p: np.ndarray) -> np.ndarray:
    """Gyroid scalar field.

    Parameters
    ----------
    p : (..., 3) array of points.

    Returns
    -------
    (...,) array of phi values.
    """
    p = np.asarray(p, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return (np.sin(x) * np.cos(y)
            + np.sin(y) * np.cos(z)
            + np.sin(z) * np.cos(x))


def grad_phi(p: np.ndarray) -> np.ndarray:
    """Analytic gradient of :func:`phi` (spec 1.1).

    Parameters
    ----------
    p : (..., 3) array of points.

    Returns
    -------
    (..., 3) array of gradient vectors.
    """
    p = np.asarray(p, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    g = np.empty_like(p)
    g[..., 0] = np.cos(x) * np.cos(y) - np.sin(z) * np.sin(x)
    g[..., 1] = -np.sin(x) * np.sin(y) + np.cos(y) * np.cos(z)
    g[..., 2] = -np.sin(y) * np.sin(z) + np.cos(z) * np.cos(x)
    return g


def normals(p: np.ndarray) -> np.ndarray:
    """Unit surface normals (normalized gradient) at points ``p``."""
    g = grad_phi(p)
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return g / n


def project_to_surface(p: np.ndarray, iters: int = 3,
                       max_step: float = 0.5) -> np.ndarray:
    """Newton-reproject points onto ``phi == 0`` (spec stage A / D).

        x <- x - phi(x) * grad_phi(x) / |grad_phi(x)|^2

    A few iterations converge quadratically near the surface.  The step is
    clamped (and suppressed where the gradient nearly vanishes, i.e. near
    critical points of ``phi``) so the iteration never shoots a point far off
    into a pore -- the source of divergence on this oscillatory field.
    """
    p = np.array(p, dtype=float, copy=True)
    for _ in range(iters):
        f = phi(p)
        g = grad_phi(p)
        gn = np.linalg.norm(g, axis=-1)
        # suppress steps where the gradient is too small to trust
        safe = gn > 1e-3
        gg = np.where(gn < 1e-12, 1.0, gn * gn)
        step = (f / gg)[..., None] * g
        # clamp step length
        slen = np.linalg.norm(step, axis=-1, keepdims=True)
        scale = np.where(slen > max_step, max_step / (slen + 1e-12), 1.0)
        step = step * scale
        step = np.where(safe[..., None], step, 0.0)
        p = p - step
    return p


def surface_residual(p: np.ndarray) -> float:
    """max |phi| over points -- used by acceptance checks."""
    return float(np.max(np.abs(phi(p)))) if len(p) else 0.0
