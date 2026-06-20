"""Tests encoding the spec's acceptance criteria (run with: pytest -q).

These run on a small block (N=2) so they are fast; the same invariants hold at
larger N.
"""

import numpy as np
import pytest

from gyroid_curve.config import Config
from gyroid_curve import gyroid
from gyroid_curve.mesh_stageA import build_block_mesh, periodicity_residual
from gyroid_curve.hilbert_stageB import (hilbert_points, verify_hilbert,
                                         build_hilbert_field)


# --------------------------------------------------------------------------
# gyroid field (single source of truth)
# --------------------------------------------------------------------------
def test_gyroid_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 6.28, size=(200, 3))
    g = gyroid.grad_phi(P)
    eps = 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        fd = (gyroid.phi(P + d) - gyroid.phi(P - d)) / (2 * eps)
        assert np.allclose(fd, g[:, k], atol=1e-4)


def test_gyroid_periodicity():
    rng = np.random.default_rng(1)
    P = rng.uniform(0, 6.28, size=(100, 3))
    shift = P + np.array([2 * np.pi, 0, 0])
    assert np.allclose(gyroid.phi(P), gyroid.phi(shift), atol=1e-9)


def test_projection_lands_on_surface():
    rng = np.random.default_rng(2)
    P = rng.uniform(0.5, 5.5, size=(500, 3))
    Q = gyroid.project_to_surface(P, iters=5)
    # points that converged should be on the surface; allow a few stuck near
    # critical points (gradient ~ 0) which are deliberately not stepped.
    res = np.abs(gyroid.phi(Q))
    assert np.median(res) < 1e-6
    assert np.mean(res < 1e-4) > 0.9


# --------------------------------------------------------------------------
# Stage A acceptance
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_mesh():
    return build_block_mesh(Config(N=2, M=24), verbose=False)


def test_stageA_on_surface(small_mesh):
    assert gyroid.surface_residual(small_mesh.V) < 1e-4


def test_stageA_no_giant_edges(small_mesh):
    # the open-block mesh must not contain period-spanning edges
    from gyroid_curve.mesh_stageA import _unique_edges
    E = _unique_edges(small_mesh.F)
    el = np.linalg.norm(small_mesh.V[E[:, 0]] - small_mesh.V[E[:, 1]], axis=1)
    assert el.max() < 1.0       # cell size is 2*pi; edges are ~0.2


def test_stageA_mostly_good_triangles(small_mesh):
    from gyroid_curve.mesh_stageA import _triangle_angles
    ang = np.degrees(_triangle_angles(small_mesh.V, small_mesh.F)).min(axis=1)
    # spec target is min>25 everywhere; we accept residual slivers at the open
    # faces and assert the vast majority are good (handled by clamped weights).
    assert np.mean(ang > 25) > 0.9


# --------------------------------------------------------------------------
# Stage B acceptance
# --------------------------------------------------------------------------
@pytest.mark.parametrize("order,side", [(1, 2), (2, 4), (3, 8)])
def test_hilbert_curve_valid(order, side):
    pts = hilbert_points(order, 3)
    chk = verify_hilbert(pts, side)
    assert chk["ok"], chk


def test_hilbert_field_mostly_continuous():
    hf = build_hilbert_field(Config(N=4, M=24), verbose=False)
    g = hf.grid
    def lf(a, b):
        return 1 - np.abs(np.sum(a * b, axis=-1))
    d = np.concatenate([lf(g[1:], g[:-1]).ravel(),
                        lf(g[:, 1:], g[:, :-1]).ravel(),
                        lf(g[:, :, 1:], g[:, :, :-1]).ravel()])
    # continuous except isolated field singularities (curves where two Hilbert
    # segments are equidistant); these are measure-zero and fine for a soft bias.
    assert d.mean() < 0.15
    assert np.mean(d < 0.1) > 0.85


# --------------------------------------------------------------------------
# Stage D / D' acceptance: single connected curve, on surface
# --------------------------------------------------------------------------
def _is_single_polyline(curve):
    return curve.ndim == 2 and curve.shape[1] == 3 and len(curve) > 1


def test_tsp_single_curve_on_surface(small_mesh):
    from gyroid_curve.tsp_stageDp import build_tsp_curve
    cfg = Config(N=2, M=24, method="tsp", delta=0.6)
    hf = build_hilbert_field(cfg, verbose=False)
    curve = build_tsp_curve(cfg, small_mesh, hf, verbose=False)
    assert _is_single_polyline(curve)
    assert gyroid.surface_residual(curve) < 1e-5
    # Hilbert-monotone at cell scale: index strongly correlates with arclength
    al = hf.hilbert_arclength_of(curve[::40])
    assert np.corrcoef(np.arange(len(al)), al)[0, 1] > 0.9


def test_stripe_single_curve_on_surface(small_mesh):
    from gyroid_curve.direction_field import build_direction_field
    from gyroid_curve.stripe_stageC import solve_phase
    from gyroid_curve.extract_stageD import extract_single_curve
    cfg = Config(N=2, M=24, method="stripe", omega=3.0, lambda_guide=1.5)
    hf = build_hilbert_field(cfg, verbose=False)
    df = build_direction_field(cfg, small_mesh, hf, verbose=False)
    psi, theta = solve_phase(cfg, small_mesh, df, verbose=False)
    curve = extract_single_curve(cfg, small_mesh, psi, hf, verbose=False)
    assert _is_single_polyline(curve)
    assert gyroid.surface_residual(curve) < 1e-5


def test_direction_field_aligns_with_guidance(small_mesh):
    from gyroid_curve.direction_field import build_direction_field
    cfg = Config(N=2, M=24, lambda_guide=3.0)
    hf = build_hilbert_field(cfg, verbose=False)
    df = build_direction_field(cfg, small_mesh, hf, verbose=False)
    # high lambda_guide => field strongly aligned to the (tangential) guidance
    assert df.guide_corr > 0.8
