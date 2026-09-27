# Dyson Sphere Design

Simulation software for the stability and construction of Dyson swarms.

## Stability maps

`solar_system_stability_map.py` computes the MEGNO chaos indicator for a massless
test particle (a swarm element) perturbed by the eight planets, integrated with
REBOUND's WHFast. Two modes:

- **(a, i)**: semimajor axis vs inclination at fixed sail lightness beta.
- **(a, beta)**: semimajor axis vs sail lightness at fixed inclination
  (`python solar_system_stability_map.py ab`).

Sail lightness beta is the ratio of radiation pressure to solar gravity; the
element then moves in the effective potential -GM(1-beta)/r and mean-motion
resonances shift inward by (1-beta)^(1/3).

```
pip install -r requirements.txt
python solar_system_stability_map.py
python -m pytest tests/
```

Parameters are set in `CONFIG` / `CONFIG_AB` at the top of the script.

## Numerical notes

- Test particle elements are heliocentric.
- For beta > 0 the radiation kick is not a small perturbation to the Kepler
  drift (relative size beta/(1-beta)), so the timestep shrinks with beta to
  keep the Wisdom-Holman splitting error near 1e-3 (see `choose_timestep`).
  High beta close to the Sun is therefore expensive.
- The script aborts if the radiation-force callback fails, and the startup
  diagnostic checks that a circular sail orbit keeps its radius.

## Known limitations

- 5000-yr integrations detect mean-motion resonance chaos and close
  encounters, but not secular resonances (nu5, nu6, nu16; 1e4-1e6 yr).
- No Poynting-Robertson drag. Its inspiral timescale,
  c r^2 / (4 G M beta) ~ 400 yr (r/AU)^2 / beta, is shorter than the
  integration for high-beta passive elements.
- MEGNO > 2 is used as the chaos threshold without a noise tolerance.
