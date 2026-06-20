"""Single configuration object for the gyroid surface-filling-curve pipeline.

All stages read their parameters from one :class:`Config` instance so the run is
fully reproducible from a single object (see spec section 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math


@dataclass
class Config:
    # --- block geometry ---------------------------------------------------
    N: int = 4          # cells per side. MUST be a power of two (Hilbert order).
    M: int = 48         # marching-cubes samples per unit cell per axis.

    # --- meshing (Stage A) ------------------------------------------------
    target_edge_len: float = 0.0     # 0 => auto (~ cell_size / M * 1.5)
    smooth_iters: int = 10           # tangential Laplacian + reproject passes
    newton_iters: int = 3            # analytic reprojection iterations / move

    # --- method selection -------------------------------------------------
    method: str = "stripe"           # "stripe" | "tsp"

    # --- stripe method (Stages C/D) --------------------------------------
    omega: float = 2.0               # stripe frequency = meander density
    lambda_guide: float = 1.0        # Hilbert alignment strength of dir. field
    dirfield_iters: int = 50         # smoothing iterations for direction field
    curve_along_hilbert: bool = True # curve runs *along* Hilbert dir (vs across)

    # --- tsp method (Stage D') -------------------------------------------
    delta: float = 0.0               # blue-noise radius. 0 => auto (~0.45)
    tsp_two_opt_passes: int = 6

    # --- misc -------------------------------------------------------------
    seed: int = 0
    out_dir: str = "output"
    name: str = "gyroid_curve"

    def __post_init__(self) -> None:
        if self.N < 1 or (self.N & (self.N - 1)) != 0:
            raise ValueError(f"N must be a power of two, got {self.N}")
        if self.method not in ("stripe", "tsp"):
            raise ValueError(f"method must be 'stripe' or 'tsp', got {self.method}")

    # --- derived quantities ----------------------------------------------
    @property
    def hilbert_order(self) -> int:
        """p such that 2**p == N."""
        return int(round(math.log2(self.N)))

    @property
    def cell_size(self) -> float:
        """World-space width of one unit cell."""
        return 2.0 * math.pi

    @property
    def block_size(self) -> float:
        """World-space width of the whole N x N x N block."""
        return 2.0 * math.pi * self.N

    @property
    def auto_target_edge_len(self) -> float:
        if self.target_edge_len > 0:
            return self.target_edge_len
        return (self.cell_size / self.M) * 1.5

    @property
    def auto_delta(self) -> float:
        if self.delta > 0:
            return self.delta
        return 0.45

    # --- serialization ----------------------------------------------------
    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls(**json.load(fh))


# A few ready-made presets.
PRESETS = {
    "preview": Config(N=2, M=24, omega=2.0, lambda_guide=1.0, name="gyroid_preview"),
    "default": Config(N=4, M=48, omega=2.0, lambda_guide=1.0, name="gyroid_curve"),
    "fine":    Config(N=4, M=64, omega=3.0, lambda_guide=1.5, name="gyroid_fine"),
}
