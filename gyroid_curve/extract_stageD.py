"""Stage D -- single-curve extraction from the global phase.

The stripe isoline ``arg(psi) = c0`` is extracted per triangle: with ``psi``
interpolated linearly, ``g = Im(psi e^{-i c0})`` is linear, so ``g = 0`` is a
straight chord; the portion with ``Re(psi e^{-i c0}) > 0`` is the ``arg = c0``
stripe (the other half is ``arg = c0 + pi``).  This yields a soup of short
segments that already join across shared edges into continuous stripe loops --
the space-filling texture, continuous by construction (no routing).

Generically this is several loops.  We then connect them into ONE curve with a
minimal set of short on-surface bridges between nearest loop endpoints, splicing
in Hilbert-position order so the single result still progresses block-globally in
Hilbert order (spec 2.D).  Every sample is reprojected onto ``phi == 0``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import gyroid
from .config import Config
from .mesh_stageA import BlockMesh
from .hilbert_stageB import HilbertField


# --------------------------------------------------------------------------
# Per-triangle isoline segments
# --------------------------------------------------------------------------
def _extract_segments(V: np.ndarray, F: np.ndarray, psi: np.ndarray,
                      c0: float = 0.0) -> np.ndarray:
    """Return an (n_seg, 2, 3) array of isoline segment endpoints."""
    z = psi * np.exp(-1j * c0)
    g = np.imag(z)        # zero set = isoline
    r = np.real(z)        # keep r > 0 half (arg = c0, not c0 + pi)

    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    ga, gb, gc = g[a], g[b], g[c]
    ra, rb, rc = r[a], r[b], r[c]
    Va, Vb, Vc = V[a], V[b], V[c]

    edges = [(ga, gb, ra, rb, Va, Vb),
             (gb, gc, rb, rc, Vb, Vc),
             (gc, ga, rc, ra, Vc, Va)]

    # crossing point and r-value on each edge where g changes sign
    pts = np.full((len(F), 3, 3), np.nan)
    rval = np.full((len(F), 3), np.nan)
    has = np.zeros((len(F), 3), bool)
    for k, (g0, g1, r0, r1, P0, P1) in enumerate(edges):
        cross = (g0 * g1) < 0
        s = np.where(cross, g0 / (g0 - g1 + 1e-300), 0.0)
        pts[:, k, :] = P0 + s[:, None] * (P1 - P0)
        rval[:, k] = r0 + s * (r1 - r0)
        has[:, k] = cross

    ncross = has.sum(axis=1)
    sel = np.where(ncross == 2)[0]            # generic case: a single chord
    if len(sel) == 0:
        return np.empty((0, 2, 3))

    segs = []
    for f in sel:
        ks = np.where(has[f])[0]
        P = pts[f, ks]          # (2,3)
        R = rval[f, ks]         # (2,)
        # clip chord to the r > 0 portion
        if R[0] >= 0 and R[1] >= 0:
            segs.append(P)
        elif R[0] < 0 and R[1] < 0:
            continue
        else:
            # one endpoint has r<0; clip at r=0
            t = R[0] / (R[0] - R[1])
            mid = P[0] + t * (P[1] - P[0])
            if R[0] >= 0:
                segs.append(np.stack([P[0], mid]))
            else:
                segs.append(np.stack([mid, P[1]]))
    if not segs:
        return np.empty((0, 2, 3))
    return np.stack(segs)


# --------------------------------------------------------------------------
# Segment soup -> polylines
# --------------------------------------------------------------------------
def _link_segments(segs: np.ndarray, weld_tol: float) -> list[np.ndarray]:
    """Weld coincident endpoints and trace into polylines (loops / arcs)."""
    if len(segs) == 0:
        return []
    endpoints = segs.reshape(-1, 3)
    tree = cKDTree(endpoints)
    # union-find weld of near-coincident endpoints
    parent = np.arange(len(endpoints))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = tree.query_pairs(weld_tol, output_type="ndarray")
    for u, v in pairs:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    node = np.array([find(x) for x in range(len(endpoints))])
    uniq, node = np.unique(node, return_inverse=True)
    coords = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(coords, node, endpoints)
    np.add.at(cnt, node, 1.0)
    coords /= cnt[:, None]

    # build adjacency: each segment connects node[2k] and node[2k+1]
    n_seg = len(segs)
    a = node[0::2]
    b = node[1::2]
    from collections import defaultdict
    adj = defaultdict(list)
    for k in range(n_seg):
        if a[k] == b[k]:
            continue
        adj[a[k]].append(b[k])
        adj[b[k]].append(a[k])

    visited_edge = set()

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    polylines = []
    # trace from degree-1 nodes first (open arcs), then remaining loops
    nodes_by_deg = sorted(adj.keys(), key=lambda n: len(adj[n]))
    starts = [n for n in nodes_by_deg if len(adj[n]) == 1] + nodes_by_deg
    for start in starts:
        for nb in adj[start]:
            if edge_key(start, nb) in visited_edge:
                continue
            # walk
            path = [start]
            prev, cur = start, nb
            visited_edge.add(edge_key(prev, cur))
            path.append(cur)
            while True:
                nxts = [x for x in adj[cur]
                        if edge_key(cur, x) not in visited_edge]
                if not nxts:
                    break
                # prefer continuing straight (not back to prev)
                nxt = nxts[0]
                for cand in nxts:
                    if cand != prev:
                        nxt = cand
                        break
                visited_edge.add(edge_key(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
                if cur == start:
                    break
            if len(path) >= 2:
                polylines.append(coords[path])
    return polylines


# --------------------------------------------------------------------------
# Connect polylines into ONE curve, ordered by Hilbert position
# --------------------------------------------------------------------------
def _connect_one_curve(polylines: list[np.ndarray], hfield: HilbertField,
                       hilbert_bias: float = 1.0
                       ) -> tuple[np.ndarray, int, float]:
    """Splice all polylines into one curve that progresses in Hilbert order with
    short bridges (spec 2.D).

    With the phase advancing along the Hilbert direction, the theta-isolines are
    stacked in Hilbert order, so we attach polylines in increasing Hilbert
    position while, among the not-yet-used polylines near the current Hilbert
    front, choosing the one whose endpoint is spatially nearest the running tail.
    This keeps bridges short *and* the macroscopic sweep Hilbert-monotone.

    ``hilbert_bias`` weights forward Hilbert progress against bridge length
    (the connection-side counterpart of ``lambda_guide``).
    """
    if not polylines:
        return np.empty((0, 3)), 0, 0.0
    n = len(polylines)

    ep = np.zeros((2 * n, 3))                  # two endpoints per polyline
    owner = np.zeros(2 * n, dtype=int)
    which = np.zeros(2 * n, dtype=int)         # 0 = start, 1 = end
    for p in range(n):
        ep[2 * p] = polylines[p][0]
        ep[2 * p + 1] = polylines[p][-1]
        owner[2 * p] = owner[2 * p + 1] = p
        which[2 * p] = 0
        which[2 * p + 1] = 1
    ep_hil = hfield.hilbert_arclength_of(ep)
    cents = np.array([pl.mean(axis=0) for pl in polylines])
    cent_hil = hfield.hilbert_arclength_of(cents)
    tree = cKDTree(ep)

    used = np.zeros(n, bool)
    start = int(np.argmin(cent_hil))
    out = [polylines[start]]
    used[start] = True
    tail = polylines[start][-1]
    tail_hil = ep_hil[2 * start + 1]
    bridges = 0
    bridge_len_sum = 0.0

    for _ in range(n - 1):
        # candidate set: spatially nearby endpoints; pick the one minimising
        # bridge length + hilbert_bias * forward-Hilbert gap (penalise going back)
        k = 24
        best = -1
        best_cost = np.inf
        while True:
            d, idx = tree.query(tail, k=min(k, 2 * n))
            idx = np.atleast_1d(idx)
            d = np.atleast_1d(d)
            for dist, e in zip(d, idx):
                p = owner[e]
                if used[p]:
                    continue
                gap = ep_hil[e] - tail_hil
                # penalise backward Hilbert moves more than forward ones
                pen = gap if gap >= 0 else -3.0 * gap
                cost = dist + hilbert_bias * pen
                if cost < best_cost:
                    best_cost = cost
                    best = e
            if best >= 0 or k >= 2 * n:
                break
            k *= 2
        if best < 0:
            break
        p = owner[best]
        pl = polylines[p]
        if which[best] == 1:
            pl = pl[::-1]
        bridges += 1
        bridge_len_sum += float(np.linalg.norm(pl[0] - tail))
        out.append(pl)
        used[p] = True
        tail = pl[-1]
        tail_hil = ep_hil[2 * p + (0 if which[best] == 1 else 1)]

    curve = np.vstack(out)
    mean_bridge = bridge_len_sum / max(1, bridges)
    return curve, bridges, mean_bridge


def extract_single_curve(cfg: Config, mesh: BlockMesh, psi: np.ndarray,
                         hfield: HilbertField, verbose: bool = True
                         ) -> np.ndarray:
    V, F = mesh.V, mesh.F
    edge_len = cfg.auto_target_edge_len
    segs = _extract_segments(V, F, psi, c0=0.0)
    if verbose:
        print(f"[D] extracted {len(segs)} isoline segments")
    polylines = _link_segments(segs, weld_tol=edge_len * 0.25)
    if verbose:
        total = sum(len(p) for p in polylines)
        print(f"[D] linked into {len(polylines)} polylines ({total} pts)")
    curve, bridges, mean_bridge = _connect_one_curve(polylines, hfield)
    curve = gyroid.project_to_surface(curve, iters=2)
    if verbose:
        print(f"[D] single curve: {len(curve)} pts, {bridges} bridges "
              f"(mean bridge {mean_bridge:.3f}), "
              f"max|phi|={gyroid.surface_residual(curve):.2e}")
    return curve
