#!/usr/bin/env python3
"""CLI entry point for the gyroid surface-filling-curve pipeline.

Examples
--------
    # default stripe-texture curve on a 4x4x4 block
    python run_gyroid.py --method stripe --N 4 --omega 3 --lambda-guide 2

    # TSP curve with a strong, clearly visible Hilbert sweep
    python run_gyroid.py --method tsp --N 4 --delta 0.5

    # quick preview
    python run_gyroid.py --preset preview

See ``gyroid_curve/README.md`` for the full description of the two knobs
(``omega``/``delta`` = density, ``lambda_guide`` = Hilbert-alignment strength).
"""

from __future__ import annotations

import argparse

from gyroid_curve.config import Config, PRESETS
from gyroid_curve.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="start from a named preset (preview|default|fine)")
    p.add_argument("--method", choices=["stripe", "tsp"], default=None)
    p.add_argument("--N", type=int, default=None, help="cells per side (power of 2)")
    p.add_argument("--M", type=int, default=None, help="mesh res per cell")
    p.add_argument("--omega", type=float, default=None, help="stripe density")
    p.add_argument("--delta", type=float, default=None, help="TSP blue-noise radius")
    p.add_argument("--lambda-guide", dest="lambda_guide", type=float,
                   default=None, help="Hilbert alignment strength")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-render", action="store_true",
                   help="skip the matplotlib PNG (still writes html/obj/ply/csv)")
    args = p.parse_args()

    cfg = PRESETS[args.preset] if args.preset else Config()
    overrides = {k: v for k, v in dict(
        method=args.method, N=args.N, M=args.M, omega=args.omega,
        delta=args.delta, lambda_guide=args.lambda_guide, seed=args.seed,
        name=args.name, out_dir=args.out_dir).items() if v is not None}
    cfg = Config(**{**cfg.__dict__, **overrides})

    result = run_pipeline(cfg, render=not args.no_render)
    cfg.to_json(f"{cfg.out_dir}/{cfg.name}_config.json")
    print("outputs:")
    for k, v in result["paths"].items():
        print(f"  {k:9s} {v}")


if __name__ == "__main__":
    main()
