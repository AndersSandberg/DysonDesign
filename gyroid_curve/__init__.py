"""Homogeneous surface-filling curve on a gyroid block.

One global gyroid field -> one periodic mesh -> one global stripe/phase field
biased by a Hilbert guidance field -> one connected isoline (the space-filling
curve).  Two frequencies, one curve (see ``gyroid_spacefilling_spec.md``).
"""

from .config import Config, PRESETS

__all__ = ["Config", "PRESETS", "run_pipeline"]


def run_pipeline(*args, **kwargs):
    # imported lazily to keep ``import gyroid_curve`` light
    from .pipeline import run_pipeline as _run
    return _run(*args, **kwargs)
