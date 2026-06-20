"""Stage B -- Hilbert guidance field.

The 3-D Hilbert curve is DEMOTED from router to guidance field (spec 0, 2.B).
We build a low-frequency unit-vector field ``H(x, y, z)`` over the block whose
direction follows the local Hilbert progression.  Downstream (Stage C) this only
*biases* the stripe direction field; it never concatenates curve segments.

Implementation:
  1. A correct 3-D Hilbert ordering of the N x N x N cell grid (Skilling's
     transpose algorithm), verified by unit-Manhattan-step adjacency.
  2. A polyline through the cell centres in Hilbert order.
  3. ``H`` sampled on a coarse grid as the (normalized) tangent of the nearest
     Hilbert segment, then Gaussian-blurred over ~1 cell width so it is a
     continuous low-frequency field, and trilinearly interpolated to any query.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import Config


# --------------------------------------------------------------------------
# 3-D Hilbert ordering (Skilling 2004 transpose algorithm)
# --------------------------------------------------------------------------
def _transpose_to_axes(X: list[int], bits: int, n: int) -> list[int]:
    """In-place Skilling transpose -> axes conversion."""
    N = 2 << (bits - 1)
    # Gray decode
    t = X[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        X[i] ^= X[i - 1]
    X[0] ^= t
    # Undo excess work
    Q = 2
    while Q != N:
        P = Q - 1
        for i in range(n - 1, -1, -1):
            if X[i] & Q:
                X[0] ^= P
            else:
                t = (X[0] ^ X[i]) & P
                X[0] ^= t
                X[i] ^= t
        Q <<= 1
    return X


def _index_to_transpose(h: int, bits: int, n: int) -> list[int]:
    """Distribute the bits of linear index ``h`` round-robin into n channels."""
    X = [0] * n
    for i in range(n * bits):
        bit = (h >> (n * bits - 1 - i)) & 1
        dim = i % n
        pos = bits - 1 - (i // n)
        X[dim] |= bit << pos
    return X


def hilbert_points(order: int, n: int = 3) -> np.ndarray:
    """All grid points of the ``2**order`` per-axis Hilbert curve in order.

    Returns an ``(2**(order*n), n)`` integer array of cell coordinates.
    """
    if order == 0:
        return np.zeros((1, n), dtype=np.int64)
    total = 1 << (order * n)
    pts = np.empty((total, n), dtype=np.int64)
    for h in range(total):
        X = _index_to_transpose(h, order, n)
        _transpose_to_axes(X, order, n)
        pts[h] = X
    return pts


def verify_hilbert(pts: np.ndarray, side: int) -> dict:
    """Acceptance check: every cell visited once, unit Manhattan steps."""
    n = pts.shape[1]
    total = side ** n
    unique = len(np.unique(pts, axis=0)) == total == len(pts)
    steps = np.abs(np.diff(pts, axis=0)).sum(axis=1)
    unit_steps = bool(np.all(steps == 1))
    in_range = bool(pts.min() >= 0 and pts.max() == side - 1)
    return {"all_unique": unique, "unit_steps": unit_steps,
            "in_range": in_range, "ok": unique and unit_steps and in_range}


# --------------------------------------------------------------------------
# Smooth guidance field H
# --------------------------------------------------------------------------
@dataclass
class HilbertField:
    centres: np.ndarray      # (n_cells, 3) Hilbert-ordered cell centres (world)
    grid: np.ndarray         # (G, G, G, 3) smoothed unit vectors
    G: int                   # samples per axis of the field grid
    period: float            # block size
    arclen: np.ndarray       # (n_cells,) cumulative Hilbert arclength at centres

    # --- query ---
    def sample(self, P: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate H at world points P -> unit vectors."""
        P = np.asarray(P, float)
        u = (P / self.period) * (self.G - 1)
        u = np.clip(u, 0, self.G - 1 - 1e-6)
        i0 = np.floor(u).astype(int)
        f = u - i0
        i1 = i0 + 1
        g = self.grid
        out = np.zeros((len(P), 3))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                         * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                         * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                    idx = (np.where(dx == 0, i0[:, 0], i1[:, 0]),
                           np.where(dy == 0, i0[:, 1], i1[:, 1]),
                           np.where(dz == 0, i0[:, 2], i1[:, 2]))
                    out += w[:, None] * g[idx]
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        nrm = np.where(nrm < 1e-9, 1.0, nrm)
        return out / nrm

    def hilbert_arclength_of(self, P: np.ndarray) -> np.ndarray:
        """For world points P, the cumulative Hilbert arclength of the nearest
        segment -- used to verify macroscopic Hilbert progression (Stage D)."""
        P = np.asarray(P, float)
        seg_a = self.centres[:-1]
        seg_b = self.centres[1:]
        d = seg_b - seg_a
        L2 = np.sum(d * d, axis=1)
        out = np.empty(len(P))
        for k in range(len(P)):
            t = np.clip(np.sum((P[k] - seg_a) * d, axis=1) / L2, 0, 1)
            proj = seg_a + t[:, None] * d
            dist = np.sum((proj - P[k]) ** 2, axis=1)
            s = int(np.argmin(dist))
            out[k] = self.arclen[s] + t[s] * (self.arclen[s + 1] - self.arclen[s])
        return out


def _nearest_segment_tangent(grid_pts: np.ndarray, seg_a: np.ndarray,
                             seg_b: np.ndarray) -> np.ndarray:
    """For each grid point, the unit tangent of the nearest Hilbert segment."""
    d = seg_b - seg_a
    L2 = np.sum(d * d, axis=1)
    dn = d / (np.sqrt(L2)[:, None] + 1e-12)
    out = np.empty((len(grid_pts), 3))
    # chunk to bound memory
    chunk = max(1, 200000 // max(1, len(seg_a)))
    for start in range(0, len(grid_pts), chunk):
        P = grid_pts[start:start + chunk]              # (c, 3)
        t = np.clip(
            np.einsum('cij,ij->ci', P[:, None, :] - seg_a[None], d) / L2, 0, 1)
        proj = seg_a[None] + t[..., None] * d[None]    # (c, S, 3)
        dist = np.sum((proj - P[:, None, :]) ** 2, axis=2)
        nearest = np.argmin(dist, axis=1)
        out[start:start + chunk] = dn[nearest]
    return out


def build_hilbert_field(cfg: Config, samples_per_cell: int = 6,
                        verbose: bool = True) -> HilbertField:
    """Run Stage B end to end."""
    order = cfg.hilbert_order
    cells = hilbert_points(order, 3)
    chk = verify_hilbert(cells, cfg.N)
    if not chk["ok"]:
        raise RuntimeError(f"Hilbert curve failed verification: {chk}")
    if verbose:
        print(f"[B] Hilbert order {order} ({len(cells)} cells), verified {chk['ok']}")

    centres = (cells + 0.5) * cfg.cell_size               # world cell centres
    seg = np.diff(centres, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seglen)])

    G = max(4, cfg.N * samples_per_cell)
    ax = (np.arange(G) + 0.5) / G * cfg.block_size
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing='ij')
    gp = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    tang = _nearest_segment_tangent(gp, centres[:-1], centres[1:])
    tang = tang.reshape(G, G, G, 3)

    # Blur as a LINE field, not a vector field: the Hilbert tangent reverses
    # direction at U-turns, so blurring raw arrows cancels.  Instead blur the
    # structure tensor T = v v^T (sign-invariant) over ~1 cell width and take its
    # dominant eigenvector -> a smooth, low-frequency orientation field.
    sigma = samples_per_cell * 0.6
    T = np.empty((G, G, G, 3, 3))
    for a in range(3):
        for b in range(3):
            T[..., a, b] = gaussian_filter(tang[..., a] * tang[..., b],
                                           sigma=sigma, mode='nearest')
    evals, evecs = np.linalg.eigh(T)        # ascending eigenvalues
    field = evecs[..., -1]                   # dominant eigenvector
    # Orient consistently with the (blurred) mean arrow to keep arrows coherent.
    mean_arrow = np.stack(
        [gaussian_filter(tang[..., c], sigma=sigma, mode='nearest')
         for c in range(3)], axis=-1)
    flip = np.sum(field * mean_arrow, axis=-1, keepdims=True) < 0
    field = np.where(flip, -field, field)
    nrm = np.linalg.norm(field, axis=-1, keepdims=True)
    field = field / np.where(nrm < 1e-9, 1.0, nrm)

    if verbose:
        print(f"[B] guidance field on {G}^3 grid, sigma={sigma:.1f} samples")
    return HilbertField(centres=centres, grid=field, G=G,
                        period=cfg.block_size, arclen=arclen)
