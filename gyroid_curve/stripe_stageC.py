"""Stage C.2/3 -- global stripe phase via the Knoeppel et al. 2015 eigenproblem.

Given the direction field (Stage C.1) and target frequency ``omega`` (= meander
spacing = curve density), solve for a complex phase ``psi`` per vertex whose
argument ``theta = arg(psi)`` advances at rate ``omega`` along the prescribed
direction.  The stripes (and hence the extracted curve) run perpendicular to
that advance direction.

We minimise the "twisted Dirichlet" energy (Knoeppel, Crane, Pinkall, Schroeder,
SIGGRAPH 2015):

    E(psi) = sum_{edges ij} w_ij | psi_j - e^{i phi_ij} psi_i |^2 ,
    phi_ij = omega * ( tau . (x_j - x_i) ),

where ``tau`` is the per-edge advance direction and ``w_ij`` is the cotangent
weight.  ``E`` is a Hermitian quadratic form ``psi^* A psi``; the smoothest unit
stripe field is the eigenvector of the smallest generalised eigenvalue of
``A psi = lambda M psi`` (``M`` = lumped mass).  Singularities of the stripe
pattern are exactly the zeros of ``psi`` and are handled gracefully by this
complex formulation (no branch-cut bookkeeping needed).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import Config
from .mesh_stageA import BlockMesh, _unique_edges
from .direction_field import DirectionField
from . import surface


def _advance_direction(cfg: Config, dirfield: DirectionField) -> np.ndarray:
    """Per-vertex direction along which the phase advances.

    For the single curve to *progress* in Hilbert order, the phase must advance
    ALONG the (Hilbert-aligned) direction field: then the theta-isolines are
    perpendicular sheets stacked in Hilbert order, and connecting them in
    theta/Hilbert order both sweeps the block along Hilbert and keeps bridges
    short (adjacent sheets are neighbours).  Hence ``curve_along_hilbert`` =>
    ``tau = t``.  Otherwise advance across it (``tau = n x t``), giving stripes
    that run along Hilbert with no global sweep.
    """
    t = dirfield.t
    if cfg.curve_along_hilbert:
        tau = t
    else:
        tau = np.cross(dirfield.normal, t)
    nn = np.linalg.norm(tau, axis=1, keepdims=True)
    return tau / np.where(nn < 1e-12, 1.0, nn)


def solve_phase(cfg: Config, mesh: BlockMesh, dirfield: DirectionField,
                solver: str = "linear", verbose: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (psi complex per vertex, theta=arg(psi) per vertex).

    ``solver="linear"`` (default, robust): the direction field is smooth and
    nearly integrable by design, so we solve the least-squares Poisson problem
    for the best-fit phase potential

        min_theta  sum_ij w_ij (theta_j - theta_i - phi_ij)^2   =>   L theta = b,

    giving a smooth phase whose isolines are long and clean.  Spacing stays
    uniform because the guidance field is smooth (no large non-integrable part).

    ``solver="eigen"``: the full Knoeppel complex eigenproblem (kept for
    reference / strongly non-integrable fields).
    """
    V, F = mesh.V, mesh.F
    n_v = len(V)
    L, mass = surface.cotmatrix_mass(mesh)        # L gives cotan weights w_ij

    E = _unique_edges(F)
    i, j = E[:, 0], E[:, 1]
    w = np.asarray(-L[i, j]).ravel()
    w = np.maximum(w, 0.0)

    tau = _advance_direction(cfg, dirfield)
    tau_edge = 0.5 * (tau[i] + tau[j])
    edge_vec = V[j] - V[i]
    phi_ij = cfg.omega * np.sum(tau_edge * edge_vec, axis=1)

    if solver == "eigen":
        eip = np.exp(1j * phi_ij)
        rows = np.concatenate([i, j, j, i])
        cols = np.concatenate([i, j, i, j])
        vals = np.concatenate([w.astype(complex), w.astype(complex),
                               -w * eip, -w * np.conj(eip)])
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n_v, n_v)).tocsr()
        A = 0.5 * (A + A.getH())
        M = sp.diags(mass)
        if verbose:
            print(f"[C] phase eigenproblem: {n_v} verts, omega={cfg.omega}...")
        psi = _smallest_eigvec(A, M, n_v)
        theta = np.angle(psi)
        return psi, theta

    # ---- linear least-squares Poisson (default) ----
    # b_i = sum_j w_ij phi_ij  (right-hand side = divergence of the target 1-form)
    b = np.zeros(n_v)
    np.add.at(b, i, w * phi_ij)
    np.add.at(b, j, -w * phi_ij)
    # L is PSD with constant nullspace; b is orthogonal to constants.  Pin one
    # vertex for a unique gauge.
    Lreg = (L + 1e-8 * sp.identity(n_v)).tocsr()
    if verbose:
        print(f"[C] phase linear solve: {n_v} verts, omega={cfg.omega}...")
    theta = spla.cg(Lreg, b, rtol=1e-8, maxiter=5000)[0]
    psi = np.exp(1j * theta)
    if verbose:
        # integrability residual (how far the field is from a perfect potential)
        res = np.sqrt(np.sum(w * (theta[j] - theta[i] - phi_ij) ** 2)
                      / (np.sum(w * phi_ij ** 2) + 1e-12))
        print(f"[C] phase relative non-integrability = {res:.3f} "
              f"(0 = perfect stripes)")
    return psi, theta


def _smallest_eigvec(A: sp.spmatrix, M: sp.spmatrix, n_v: int) -> np.ndarray:
    """Smallest generalised eigenvector of (A, M), robust to solver quirks."""
    # Shift-invert near 0 finds the smallest eigenvalues efficiently.
    for sigma in (-1e-6, -1e-4, -1e-2):
        try:
            vals, vecs = spla.eigsh(A, k=1, M=M, sigma=sigma, which="LM",
                                    maxiter=5000, tol=1e-7)
            return vecs[:, 0]
        except Exception:
            continue
    # Fallback: regularise and use LOBPCG with a random start.
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((n_v, 1))
         + 1j * rng.standard_normal((n_v, 1)))
    Areg = (A + 1e-8 * sp.identity(n_v)).tocsr()
    vals, vecs = spla.lobpcg(Areg, X, B=M, largest=False, maxiter=2000,
                             tol=1e-6)
    return vecs[:, 0]
