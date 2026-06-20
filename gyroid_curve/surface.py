"""Shared surface operators used by both the stripe and TSP methods.

Everything here is built on the Stage-A mesh plus the analytic ``phi`` /
``grad_phi`` (the single source of truth).  Provides:
  - area-weighted + blue-noise (Poisson-disk) surface sampling;
  - per-vertex tangent frames from analytic normals;
  - a robustified cotangent Laplacian and lumped mass (for the stripe solve).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from . import gyroid
from .mesh_stageA import BlockMesh


# --------------------------------------------------------------------------
# Surface sampling
# --------------------------------------------------------------------------
def sample_faces_uniform(mesh: BlockMesh, n: int, rng) -> np.ndarray:
    """Area-weighted uniform random points on the mesh surface."""
    V, F = mesh.V, mesh.F
    tri = V[F]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    prob = area / area.sum()
    fi = rng.choice(len(F), size=n, p=prob)
    u = rng.random(n)
    v = rng.random(n)
    over = u + v > 1
    u[over] = 1 - u[over]
    v[over] = 1 - v[over]
    P = (tri[fi, 0]
         + u[:, None] * (tri[fi, 1] - tri[fi, 0])
         + v[:, None] * (tri[fi, 2] - tri[fi, 0]))
    return P


def blue_noise_surface(mesh: BlockMesh, radius: float, rng,
                       oversample: int = 12) -> np.ndarray:
    """Poisson-disk sampling of the surface by dart elimination.

    Generate a dense uniform pool, then greedily keep points whose pairwise
    distance is >= ``radius`` (weighted sample elimination, simplified).  Points
    are analytically reprojected onto ``phi == 0``.
    """
    V, F = mesh.V, mesh.F
    tri = V[F]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area = float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())
    # approx number of disks that fit (hexagonal packing density ~0.9)
    n_target = max(16, int(0.75 * area / (np.pi * (radius / 2) ** 2)))
    pool = sample_faces_uniform(mesh, n_target * oversample, rng)

    # Greedy Poisson-disk elimination with an O(n) spatial-hash grid.
    cell = radius
    grid: dict[tuple[int, int, int], list[np.ndarray]] = {}
    keep = []
    r2 = radius * radius

    def key(p):
        return (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))

    for p in pool:
        k = key(p)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if np.sum((q - p) ** 2) < r2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            keep.append(p)
            grid.setdefault(k, []).append(p)
    S = np.array(keep)
    return gyroid.project_to_surface(S, iters=3)


# --------------------------------------------------------------------------
# Tangent frames
# --------------------------------------------------------------------------
def tangent_frames(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-point orthonormal tangent frame (e1, e2, normal) from analytic grad.

    e1 is an arbitrary but smooth-ish tangent (built from the least-aligned
    global axis), e2 = n x e1.
    """
    n = gyroid.normals(P)
    # choose helper axis least aligned with n
    ax = np.tile(np.array([1.0, 0, 0]), (len(P), 1))
    mask = np.abs(n[:, 0]) > 0.9
    ax[mask] = np.array([0.0, 1.0, 0.0])
    e1 = ax - np.sum(ax * n, axis=1, keepdims=True) * n
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(n, e1)
    return e1, e2, n


# --------------------------------------------------------------------------
# Cotangent Laplacian + mass (robustified)
# --------------------------------------------------------------------------
def cotmatrix_mass(mesh: BlockMesh, clamp: float = 1e3
                   ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Robust cotan Laplacian L (n,n, PSD) and lumped mass vector.

    Cotangents from sliver triangles are clamped to keep the operator
    well-conditioned (spec: residual slivers handled by clamped weights).
    Returns ``L`` such that the Dirichlet energy is ``x^T L x`` with positive
    off-diagonal weights ``w_ij`` stored as ``-w_ij`` off-diagonal.
    """
    V, F = mesh.V, mesh.F
    n = len(V)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    def cot(p, q, r):
        # cotangent of angle at vertex p in triangle (p,q,r)
        u = q - p
        v = r - p
        num = np.sum(u * v, axis=1)
        den = np.linalg.norm(np.cross(u, v), axis=1)
        den = np.where(den < 1e-12, 1e-12, den)
        c = num / den
        return np.clip(c, -clamp, clamp)

    c0 = cot(V[i0], V[i1], V[i2])   # angle at i0 -> weight on edge (i1,i2)
    c1 = cot(V[i1], V[i2], V[i0])   # angle at i1 -> weight on edge (i2,i0)
    c2 = cot(V[i2], V[i0], V[i1])   # angle at i2 -> weight on edge (i0,i1)

    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    W = 0.5 * np.concatenate([c0, c0, c1, c1, c2, c2])
    # clamp negative edge weights to 0 for a PSD operator (intrinsic-Delaunay-ish)
    W = np.maximum(W, 0.0)
    A = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A

    # lumped (barycentric) mass
    a = V[i1] - V[i0]
    b = V[i2] - V[i0]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    mass = np.zeros(n)
    np.add.at(mass, i0, area / 3)
    np.add.at(mass, i1, area / 3)
    np.add.at(mass, i2, area / 3)
    mass = np.where(mass < 1e-12, 1e-12, mass)
    return L.tocsr(), mass
