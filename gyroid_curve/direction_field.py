"""Stage C.1 -- global stripe direction field, biased toward the Hilbert field.

We build one unit *line* field on the whole Stage-A mesh that is (a) as smooth as
possible and (b) biased toward the surface-tangential projection of the Hilbert
guidance field ``H`` with strength ``lambda_guide`` (spec 2.C.1, 4).  This single
global coupling is what makes one curve fill locally yet progress globally.

The field is a *line* field (orientation mod pi), so neighbouring 180-degree
disagreements do not cancel.  We smooth it with the structure-tensor (outer
product) method: accumulate ``sum_j t_j t_j^T + lambda * h_i h_i^T`` over the one
ring, take the dominant eigenvector, project to the tangent plane, renormalise,
and iterate.  This avoids an explicit connection/parallel-transport solve while
still producing a smooth, guidance-aligned field.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Config
from .mesh_stageA import BlockMesh, _unique_edges
from .hilbert_stageB import HilbertField
from . import surface


@dataclass
class DirectionField:
    t: np.ndarray            # (n_v, 3) unit tangent line directions
    e1: np.ndarray
    e2: np.ndarray
    normal: np.ndarray
    guide_corr: float        # mean |t . h_tan| achieved (alignment measure)


def _project_tangent(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = v - np.sum(v * n, axis=1, keepdims=True) * n
    nn = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(nn < 1e-12, 1.0, nn)


def build_direction_field(cfg: Config, mesh: BlockMesh, hfield: HilbertField,
                          verbose: bool = True) -> DirectionField:
    V, F = mesh.V, mesh.F
    n_v = len(V)
    e1, e2, normal = surface.tangent_frames(V)

    # H projected onto each vertex tangent plane.
    H = hfield.sample(V)
    h = _project_tangent(H, normal)

    lam = cfg.lambda_guide
    E = _unique_edges(F)
    i, j = E[:, 0], E[:, 1]

    # initialise field with the (tangential) guidance direction
    t = h.copy()
    # vertices where guidance is degenerate: fall back to e1
    bad = np.linalg.norm(t, axis=1) < 1e-6
    t[bad] = e1[bad]

    # symmetric 3x3 structure tensor has 6 unique entries
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for _ in range(cfg.dirfield_iters):
        # accumulate neighbour outer products t_j t_j^T into each vertex
        acc = np.zeros((n_v, 6))
        tout = np.stack([t[:, a] * t[:, b] for a, b in pairs], axis=1)
        for c in range(6):
            np.add.at(acc[:, c], i, tout[j, c])
            np.add.at(acc[:, c], j, tout[i, c])
        # guidance term lambda * h h^T
        hout = np.stack([h[:, a] * h[:, b] for a, b in pairs], axis=1)
        acc += lam * hout

        T = np.empty((n_v, 3, 3))
        T[:, 0, 0] = acc[:, 0]; T[:, 1, 1] = acc[:, 1]; T[:, 2, 2] = acc[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = acc[:, 3]
        T[:, 0, 2] = T[:, 2, 0] = acc[:, 4]
        T[:, 1, 2] = T[:, 2, 1] = acc[:, 5]

        evals, evecs = np.linalg.eigh(T)
        t = _project_tangent(evecs[..., -1], normal)

    corr = float(np.mean(np.abs(np.sum(t * h, axis=1))))
    if verbose:
        print(f"[C] direction field: lambda_guide={lam}, "
              f"mean|t.h_tan|={corr:.3f}")
    return DirectionField(t=t, e1=e1, e2=e2, normal=normal, guide_corr=corr)
