"""Stage A -- global periodic surface mesh of the whole N x N x N block.

We evaluate ONE global gyroid field over the whole block and extract a single
isosurface, so the surface is continuous and uniformly handed by construction
(spec 0, 2.A).  No per-cell meshing, no fundamental-domain stitching.

Steps:
  1. Sample ``phi`` on an ``(MN+1)^3`` grid over ``[0, 2*pi*N]`` where the last
     slice on each axis duplicates the first (periodic wrap).  Because the field
     is identical on opposite block faces, the 2-D marching-cubes crossing
     pattern is identical there, so the opposite boundary rings coincide under
     the block-period translation.
  2. Marching cubes for starting topology (skimage).
  3. Weld the periodic seam: fold the max-face onto the min-face and merge
     coincident vertices -> a closed surface on the 3-torus, no boundary.
  4. Tangential Laplacian smoothing in periodic (minimum-image) coordinates with
     analytic Newton reprojection after every move.

Output: a :class:`BlockMesh` (vertices, faces, analytic-gradient normals).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

from . import gyroid
from .config import Config


@dataclass
class BlockMesh:
    V: np.ndarray            # (n_v, 3) vertices
    F: np.ndarray            # (n_f, 3) triangle indices
    period: float            # block size (period of the torus)

    @property
    def N(self) -> np.ndarray:
        return gyroid.normals(self.V)

    def min_angle_deg(self) -> float:
        return float(np.degrees(_triangle_angles(self.V, self.F).min()))

    def stats(self) -> dict:
        ang = np.degrees(_triangle_angles(self.V, self.F))
        return {
            "n_vertices": int(len(self.V)),
            "n_faces": int(len(self.F)),
            "min_angle_deg": float(ang.min()),
            "mean_angle_deg": float(ang.mean()),
            "max_abs_phi": gyroid.surface_residual(self.V),
        }


def _triangle_angles(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """All interior angles (radians) of every triangle, shape (n_f, 3)."""
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    angs = np.empty((len(F), 3))
    for i, (p, q, r) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
        u = q - p
        w = r - p
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
        angs[:, i] = np.arccos(np.clip(np.sum(u * w, axis=1), -1.0, 1.0))
    return angs


def _build_field(cfg: Config) -> tuple[np.ndarray, float]:
    """Sample phi on the periodic-wrapped grid. Returns (field, spacing)."""
    n = cfg.M * cfg.N
    spacing = cfg.block_size / n
    # n+1 samples; index n duplicates index 0 (periodic wrap).
    coords = np.arange(n + 1) * spacing
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    F = gyroid.phi(np.stack([X, Y, Z], axis=-1))
    return F, spacing


def _weld_periodic(V: np.ndarray, F: np.ndarray, period: float,
                   tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Fold the max block face onto the min face and merge coincident vertices.

    Vertices whose coordinate is within ``tol`` of ``period`` are snapped to 0.
    Because opposite faces carry an identical crossing pattern, the snapped
    vertices coincide exactly with their min-face partners and merge, welding the
    surface into a closed torus.
    """
    Vw = V.copy()
    near_max = np.abs(Vw - period) < tol
    Vw[near_max] = 0.0

    # Merge by quantized position.
    quant = np.round(Vw / tol).astype(np.int64)
    _, inverse = np.unique(quant, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    new_V = np.zeros((inverse.max() + 1, 3))
    counts = np.zeros(inverse.max() + 1)
    np.add.at(new_V, inverse, Vw)
    np.add.at(counts, inverse, 1.0)
    new_V /= counts[:, None]

    new_F = inverse[F]
    # Drop degenerate triangles created by welding.
    good = ((new_F[:, 0] != new_F[:, 1])
            & (new_F[:, 1] != new_F[:, 2])
            & (new_F[:, 0] != new_F[:, 2]))
    return new_V, new_F[good]


def _unique_edges(F: np.ndarray) -> np.ndarray:
    """Undirected unique edges as an (n_e, 2) array."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _min_image(d: np.ndarray, period: float) -> np.ndarray:
    """Minimum-image wrap of displacement vectors to [-period/2, period/2)."""
    return d - period * np.round(d / period)


def _face_pins(V: np.ndarray, period: float, tol: float = 1e-6):
    """Identify boundary-face vertices and the coordinate value each is pinned to.

    Returns ``pin_val`` (same shape as V, NaN where free, 0/period where the
    vertex lies on a face) so the six outer faces stay planar -- this is what
    keeps the opposite-face rings identical under the period translation.
    """
    pin_val = np.full_like(V, np.nan)
    on_lo = np.abs(V) < tol
    on_hi = np.abs(V - period) < tol
    pin_val[on_lo] = 0.0
    pin_val[on_hi] = period
    boundary = (on_lo | on_hi).any(axis=1)
    return pin_val, boundary


def _apply_pins(V: np.ndarray, pin_val: np.ndarray) -> None:
    m = ~np.isnan(pin_val)
    V[m] = pin_val[m]


def _reproject_pinned(V: np.ndarray, pin_val: np.ndarray,
                      iters: int) -> np.ndarray:
    """Newton-reproject onto phi==0 while keeping pinned face coordinates fixed,
    so face vertices stay in-plane *and* on the surface."""
    V = V.copy()
    _apply_pins(V, pin_val)
    for _ in range(iters):
        V = gyroid.project_to_surface(V, iters=1)
        _apply_pins(V, pin_val)
    return V


def _smooth_and_reproject(V: np.ndarray, F: np.ndarray, period: float,
                          iters: int, newton_iters: int,
                          alpha: float = 0.5) -> np.ndarray:
    """Tangential Laplacian smoothing in periodic coords + Newton reprojection.

    Uniform-weight Laplacian over one-ring neighbours, accumulated over edges so
    it is vectorised.  Neighbour offsets use the minimum-image convention so the
    welded torus seam is handled correctly.  The displacement is projected onto
    the tangent plane before re-snapping to ``phi == 0``.
    """
    E = _unique_edges(F)
    i, j = E[:, 0], E[:, 1]
    n_v = len(V)
    V = V.copy()
    pin_val, boundary = _face_pins(V, period)
    free = ~boundary
    for _ in range(max(0, iters)):
        # offset from each endpoint to the other (minimum image)
        dij = _min_image(V[j] - V[i], period)   # i -> j
        accum = np.zeros((n_v, 3))
        count = np.zeros(n_v)
        np.add.at(accum, i, dij)
        np.add.at(accum, j, -dij)
        np.add.at(count, i, 1.0)
        np.add.at(count, j, 1.0)
        count = np.where(count < 1, 1.0, count)
        lap = accum / count[:, None]
        nrm = gyroid.normals(V)
        # tangential component only; pin boundary-face vertices in place
        lap = lap - np.sum(lap * nrm, axis=1, keepdims=True) * nrm
        lap[boundary] = 0.0
        Vnew = V + alpha * lap
        Vnew[free] = gyroid.project_to_surface(Vnew[free], iters=newton_iters)
        V = Vnew
    return V


def periodicity_residual(mesh: BlockMesh) -> float:
    """Max mismatch of opposite-face boundary rings under the block-period
    translation (acceptance A).  For each boundary vertex on a min face, the
    nearest min-image partner on the opposite face should coincide.  Returns the
    largest such distance (0 => perfectly tiling)."""
    V = mesh.V
    P = mesh.period
    tol = 1e-4
    worst = 0.0
    for axis in range(3):
        lo = V[np.abs(V[:, axis]) < tol]
        hi = V[np.abs(V[:, axis] - P) < tol]
        if len(lo) == 0 or len(hi) == 0:
            continue
        hi_shift = hi.copy()
        hi_shift[:, axis] -= P
        tree = cKDTree(hi_shift)
        d, _ = tree.query(lo)
        worst = max(worst, float(d.max()))
    return worst


def build_block_mesh(cfg: Config, weld: bool = False,
                     verbose: bool = True) -> BlockMesh:
    """Run Stage A end to end.

    ``weld=False`` (default) keeps the block as an OPEN surface in R^3 with
    boundaries only on the six outer faces.  The opposite-face boundary rings are
    identical under the block-period translation (so the block tiles seamlessly --
    see :func:`periodicity_residual`), but we keep coordinates Euclidean so every
    downstream metric operation (edge lengths, cotan weights, isoline points) is
    correct.  ``weld=True`` folds the seam into a closed 3-torus, which is
    topologically nicer but introduces period-spanning edges that are only short
    under the periodic metric -- use only with periodic-aware downstream code.
    """
    if verbose:
        n = cfg.M * cfg.N
        print(f"[A] sampling phi on {n + 1}^3 grid "
              f"({(n + 1) ** 3 * 8 / 1e6:.0f} MB)...")
    field, spacing = _build_field(cfg)

    if verbose:
        print("[A] marching cubes...")
    V, F, _, _ = measure.marching_cubes(field, level=0.0, spacing=(spacing,) * 3)
    if verbose:
        print(f"[A] raw MC: {len(V):,} verts, {len(F):,} faces.")

    if weld:
        V, F = _weld_periodic(V, F, cfg.block_size)
        if verbose:
            print(f"[A] welded torus: {len(V):,} verts, {len(F):,} faces.")

    # Initial exact reprojection (MC verts are linear approximations).
    # Boundary-face vertices are reprojected in-plane so the six faces stay flat
    # and the opposite rings remain identical under the period translation.
    if weld:
        V = gyroid.project_to_surface(V, iters=cfg.newton_iters)
    else:
        pin_val, _ = _face_pins(V, cfg.block_size)
        V = _reproject_pinned(V, pin_val, iters=cfg.newton_iters + 2)

    if verbose:
        print(f"[A] smoothing + reprojecting...")
    V = _smooth_and_reproject(V, F, cfg.block_size,
                              iters=cfg.smooth_iters,
                              newton_iters=cfg.newton_iters)

    mesh = BlockMesh(V=V, F=F, period=cfg.block_size)
    if verbose:
        s = mesh.stats()
        print(f"[A] done: {s['n_vertices']:,} verts, {s['n_faces']:,} faces, "
              f"min angle {s['min_angle_deg']:.1f} deg, "
              f"max|phi| {s['max_abs_phi']:.2e}, "
              f"periodicity {periodicity_residual(mesh):.2e}")
    return mesh
