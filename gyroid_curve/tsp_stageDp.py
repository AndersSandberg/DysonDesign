"""Stage D' -- robust global TSP fallback (spec 2.D').

A *global* (never per-cell) method that always produces a renderable single
curve:
  1. Blue-noise (Poisson-disk) sample the ENTIRE block mesh at radius ``delta``
     -> homogeneous density via one radius.
  2. Build one OPEN-path tour over all samples, seeded/biased by Hilbert
     position so the tour sweeps the block in Hilbert order rather than
     crisscrossing the whole volume.  Greedy nearest-neighbour under a metric
     ``euclid + w*|hilbert_arclen_i - hilbert_arclen_j|`` + windowed 2-opt.
  3. Reproject the densified polyline onto ``phi == 0``.

Less elegant texture than stripes, but always works (spec 6).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import gyroid
from .config import Config
from .mesh_stageA import BlockMesh
from .hilbert_stageB import HilbertField, hilbert_points
from .surface import blue_noise_surface


def _nn_path(P: np.ndarray, start: int) -> list[int]:
    """Greedy nearest-neighbour open path over a small point set (local indices)."""
    n = len(P)
    if n <= 1:
        return list(range(n))
    tree = cKDTree(P)
    visited = np.zeros(n, bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        k = min(n, 12)
        _, idx = tree.query(P[cur], k=k)
        nxt = -1
        for j in np.atleast_1d(idx):
            if not visited[j]:
                nxt = int(j)
                break
        if nxt < 0:
            rem = np.where(~visited)[0]
            nxt = int(rem[np.argmin(np.linalg.norm(P[rem] - P[cur], axis=1))])
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order


def _two_opt_window(P: np.ndarray, order: list[int], passes: int,
                    window: int = 30) -> list[int]:
    """Windowed 2-opt on pure spatial distance to remove local crossings."""
    order = list(order)
    n = len(order)
    if n < 4:
        return order

    def d(i, j):
        return np.linalg.norm(P[order[i]] - P[order[j]])

    for _ in range(passes):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, min(n - 1, i + window)):
                if d(i - 1, j) + d(i, j + 1) + 1e-9 < d(i - 1, i) + d(j, j + 1):
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return order


def _hilbert_cell_tour(S: np.ndarray, cfg: Config,
                       passes: int) -> np.ndarray:
    """Global tour ordered by Hilbert cell, NN within each cell, adjacent
    cell-paths joined by nearest endpoints.

    Because consecutive Hilbert cells are spatially adjacent (unit Manhattan
    step), every inter-cell link is short, so the tour sweeps the block in
    Hilbert order with no long jumps -- homogeneous density (from blue noise),
    global coherence (from Hilbert seeding).
    """
    cells = hilbert_points(cfg.hilbert_order, 3)
    rank = {tuple(c): i for i, c in enumerate(cells)}

    cell_idx = np.floor(S / cfg.cell_size).astype(int)
    cell_idx = np.clip(cell_idx, 0, cfg.N - 1)

    # bucket sample indices by cell
    buckets: dict[int, list[int]] = {}
    for i, c in enumerate(cell_idx):
        r = rank[tuple(c)]
        buckets.setdefault(r, []).append(i)

    full_order: list[int] = []
    prev_endpoint = None
    for r in range(len(cells)):
        members = buckets.get(r, [])
        if not members:
            continue
        members = np.array(members)
        P = S[members]
        # start nearest to where the previous cell-path ended
        if prev_endpoint is None:
            start = 0
        else:
            start = int(np.argmin(np.linalg.norm(P - prev_endpoint, axis=1)))
        local = _nn_path(P, start)
        local = _two_opt_window(P, local, passes)
        seq = members[local].tolist()
        full_order.extend(seq)
        prev_endpoint = S[seq[-1]]
    return np.array(full_order, dtype=int)


def _densify_reproject(P: np.ndarray, max_seg: float,
                       local: float) -> tuple[np.ndarray, int]:
    """Subdivide *local* segments and reproject; leave long jumps as chords.

    Midpoints of a straight segment between two surface points only stay near the
    surface when the endpoints are close (same sheet).  Subdividing a long jump
    would push midpoints deep into a pore where Newton reprojection diverges, so
    long jumps are kept as a single chord between two (on-surface) endpoints.
    Returns the densified polyline and the count of long jumps.
    """
    out = [P[0]]
    long_jumps = 0
    for k in range(1, len(P)):
        a, b = P[k - 1], P[k]
        d = np.linalg.norm(b - a)
        if d > local:
            long_jumps += 1
            out.append(b)               # keep chord; both ends are on-surface
            continue
        steps = max(1, int(np.ceil(d / max_seg)))
        for s in range(1, steps + 1):
            out.append(a + (b - a) * (s / steps))
    out = np.array(out)
    out = gyroid.project_to_surface(out, iters=3)
    return out, long_jumps


def build_tsp_curve(cfg: Config, mesh: BlockMesh, hfield: HilbertField,
                    verbose: bool = True) -> np.ndarray:
    """Run Stage D' end to end. Returns one ordered (n,3) polyline."""
    rng = np.random.default_rng(cfg.seed)
    delta = cfg.auto_delta
    if verbose:
        print(f"[D'] blue-noise sampling, radius delta={delta:.3f}...")
    S = blue_noise_surface(mesh, delta, rng)
    if verbose:
        print(f"[D'] {len(S)} samples. building Hilbert-cell tour...")

    order = _hilbert_cell_tour(S, cfg, passes=cfg.tsp_two_opt_passes)
    P = S[order]
    curve, long_jumps = _densify_reproject(P, max_seg=delta * 0.5,
                                           local=delta * 2.5)
    if verbose:
        print(f"[D'] curve: {len(curve)} points, {long_jumps} long jumps, "
              f"max|phi|={gyroid.surface_residual(curve):.2e}")
    return curve
