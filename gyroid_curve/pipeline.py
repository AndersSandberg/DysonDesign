"""Pipeline orchestration: run the requested method end to end.

    Stage A  global periodic mesh
    Stage B  Hilbert guidance field
    Stage C  global stripe direction + phase field      (method == "stripe")
    Stage D  single-curve isoline extraction            (method == "stripe")
    Stage D' global Hilbert-biased TSP fallback          (method == "tsp")
    Stage E  export + render
"""

from __future__ import annotations

import time
import numpy as np

from .config import Config
from .mesh_stageA import build_block_mesh
from .hilbert_stageB import build_hilbert_field
from . import output_stageE


def run_pipeline(cfg: Config, render: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    if verbose:
        print(f"=== gyroid surface-filling curve: method={cfg.method}, "
              f"N={cfg.N}, M={cfg.M} ===")

    mesh = build_block_mesh(cfg, verbose=verbose)
    hfield = build_hilbert_field(cfg, verbose=verbose)

    result = {"mesh": mesh, "hfield": hfield, "config": cfg}

    if cfg.method == "tsp":
        from .tsp_stageDp import build_tsp_curve
        curve = build_tsp_curve(cfg, mesh, hfield, verbose=verbose)
        result["aux"] = {}
    else:
        from .direction_field import build_direction_field
        from .stripe_stageC import solve_phase
        from .extract_stageD import extract_single_curve
        dirfield = build_direction_field(cfg, mesh, hfield, verbose=verbose)
        psi, theta = solve_phase(cfg, mesh, dirfield, verbose=verbose)
        curve = extract_single_curve(cfg, mesh, psi, hfield, verbose=verbose)
        result["aux"] = {"dirfield": dirfield, "psi": psi, "theta": theta}

    result["curve"] = curve
    paths = output_stageE.export_all(cfg.out_dir, cfg.name, curve, mesh,
                                     render=render, verbose=verbose)
    result["paths"] = paths
    if verbose:
        print(f"=== done in {time.time() - t0:.1f}s; "
              f"curve has {len(curve)} points ===")
    return result
