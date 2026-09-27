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

## Growth model

`dyson_growth.py` is a fast throughput ODE for bootstrapping a swarm from
asteroid feedstock: self-replicating machinery in the belt turns feedstock
into collectors, which are transferred to radius r and beam power back.
It asks how long it takes to capture a fraction f of the solar luminosity,
and where the collectors should go.

```
python dyson_growth.py --f 0.1 --r 1.0 --tau 1.0 --sigma 1e-3
```

Main results, all checked in `tests/test_dyson_growth.py`:

- Balanced growth solves an Euler-Lotka equation,
  lambda t_E = (1 - lambda tau) exp(-lambda D), where tau is the machinery
  replication time, t_E the collectors' energy payback time and D the
  transit delay. To first order, 1/lambda = tau + t_E + D: the three add
  in series.
- For thin films (sigma ~ 1e-3 kg/m^2), t_E is 20-60 minutes, so energy
  never limits growth; replication time and transit delay do.
- The time-optimal policy is bang-bang: build machinery until it equals the
  target collector mass, then spend one tau building collectors
  (as in Cohen 1971 and Macevicz & Oster 1976).
  T* = tau (ln(M/K0) + 1) + D.
- Collector mass scales as r^2, but time depends on it only through
  ln(M), so the choice of radius is weak. It is set by the transit delay
  and the collectors' thermal limit, and capped by feedstock.

Figures: `figures/growth_trajectory.png`, `figures/growth_time_vs_radius.png`,
`figures/growth_feedstock_frontier.png`.

Not yet included: self-shading, Mercury as feedstock, recycling machinery
into collectors at the end, and sail-spiral transfer instead of Hohmann.

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
