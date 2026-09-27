#!/usr/bin/env python3
"""
Throughput model for bootstrapping a Dyson swarm from asteroid feedstock.

A fast ODE model: no orbital dynamics, no trade. Mass flows from the asteroid
belt through self-replicating machinery into light collectors at orbital
radius r. Collected power feeds back into the machinery.

Stocks
    K   machinery mass at the belt (mines, refineries, fabs)          [kg]
    T_j collectors in transit to r (Erlang chain of n stages)          [kg]
    C   deployed collector mass at r                                   [kg]
    X   raw feedstock mined so far (M_belt - X remains)                [kg]
    W   waste: tailings plus worn-out machinery and collectors         [kg]

Flows (per year)
    Q   = min(K / tau, P / e(r)) * g(F)      finished-product output
    u Q -> K, (1 - u) Q -> transit -> C,  Q / y raw feedstock mined
    P   = P_seed + eta_beam * eta(r) * S(r) * C / sigma

The captured fraction of solar luminosity is f = C / (sigma 4 pi r^2).
No self-shading, so f is only meaningful for f <~ 1.

Analytic results (derived in docstrings below, checked in tests/):
  * Balanced exponential growth solves the Euler-Lotka-type equation
        lambda * t_E = (1 - lambda * tau) * exp(-lambda * D)
    with t_E = e / p' the energy payback time and D the transit delay.
    For D = 0, 1/lambda = tau + t_E: replication time and energy payback
    time add in series.
  * With capital as the binding constraint, the time-optimal policy is
    bang-bang: build machinery until K equals the target collector mass,
    then spend exactly one tau building collectors
    (cf. Cohen 1971 for annual plants; Macevicz & Oster 1976 for social
    insect colonies). T* = tau (ln(M/K0) + 1) + D.

Requires: numpy, scipy, matplotlib
"""

from dataclasses import dataclass, replace
import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

# ============================================================
# Physical constants
# ============================================================
L_SUN = 3.828e26          # W
AU = 1.495978707e11       # m
GM_SUN = 1.32712440018e20 # m^3 s^-2
YR = 3.15576e7            # s
SIGMA_SB = 5.670374419e-8 # W m^-2 K^-4
S_1AU = L_SUN / (4 * np.pi * AU ** 2)  # ~1361 W/m^2


# ============================================================
# Parameters (all engineering values are assumptions to vary)
# ============================================================
@dataclass(frozen=True)
class Params:
    # Feedstock
    a_belt: float = 2.7           # AU, where mining and fabrication happen
    M_belt: float = 2.4e21        # kg, total main-belt mass
    yield_frac: float = 0.3       # useful product per kg of raw feedstock

    # Collectors
    sigma: float = 1e-3           # kg/m^2 areal density (thin-film mirror/PV)
    eta0: float = 0.3             # conversion efficiency at T_ref
    T_ref: float = 300.0          # K
    eta_temp_coeff: float = 0.004 # fractional efficiency loss per K above T_ref
    eta_beam: float = 0.5         # power beamed from collectors to industry

    # Industry
    tau: float = 1.0              # yr; machinery processes its own mass in tau
    e_process: float = 1e8        # J/kg for mining, refining and fabrication
    eta_transport: float = 0.5    # mass-driver efficiency for the transfer delta-v
    P_seed: float = 1e7           # W, initial power supply
    K0: float = 1e6               # kg, seed machinery

    # Lifetimes (np.inf disables wear)
    life_K: float = np.inf        # yr
    life_C: float = np.inf        # yr

    # Numerics
    n_transit: int = 4            # Erlang stages for the transit delay
    D_min: float = 0.02           # yr, deployment time for collectors built in situ


# ============================================================
# Physics of a collector at radius r [AU]
# ============================================================
def flux(r):
    """Solar flux at r [W/m^2]."""
    return S_1AU / r ** 2


def collector_temperature(r):
    """
    Equilibrium temperature of a flat plate facing the Sun, absorbing on one
    side and radiating from both: T = (S / (2 sigma_SB))^(1/4), ~331 K at 1 AU.
    """
    return (flux(r) / (2 * SIGMA_SB)) ** 0.25


def efficiency(r, p):
    """Conversion efficiency, falling linearly with temperature (PV-like)."""
    eta = p.eta0 * (1 - p.eta_temp_coeff * (collector_temperature(r) - p.T_ref))
    return float(np.clip(eta, 0.0, p.eta0))


def specific_power(r, p):
    """Power delivered to industry per kg of deployed collector [W/kg]."""
    return p.eta_beam * efficiency(r, p) * flux(r) / p.sigma


def hohmann_dv(r1, r2):
    """Total delta-v [m/s] for a Hohmann transfer between circular orbits (AU)."""
    r1m, r2m = r1 * AU, r2 * AU
    v1, v2 = np.sqrt(GM_SUN / r1m), np.sqrt(GM_SUN / r2m)
    s = r1m + r2m
    return abs(v1 * (np.sqrt(2 * r2m / s) - 1)) + abs(v2 * (1 - np.sqrt(2 * r1m / s)))


def transit_time(r, p):
    """Hohmann transfer time from the belt to r [yr]."""
    if np.isclose(r, p.a_belt):
        return p.D_min
    return max(p.D_min, 0.5 * ((r + p.a_belt) / 2) ** 1.5)


def energy_per_kg(r, p):
    """Energy to mine, make and deliver one kg of collector to r [J/kg]."""
    return p.e_process + hohmann_dv(p.a_belt, r) ** 2 / (2 * p.eta_transport)


def energy_payback_time(r, p):
    """
    Time for a collector at r to deliver the energy that made it [yr].
    Infinite where the collector is too hot to work.
    """
    sp = specific_power(r, p)
    return np.inf if sp <= 0 else energy_per_kg(r, p) / sp / YR


def target_mass(r, f, p):
    """Collector mass that captures a fraction f of the Sun's output at r [kg]."""
    return f * p.sigma * 4 * np.pi * (r * AU) ** 2


def feedstock_limit(r, p):
    """
    Largest capturable fraction if all usable feedstock became collectors.
    Machinery also needs mass, so the practical limit is lower (see
    analytic_time_to_target).
    """
    return p.yield_frac * p.M_belt / target_mass(r, 1.0, p)


# ============================================================
# Analytic growth results
# ============================================================
def balanced_growth_rate(r, p):
    """
    Exponential growth rate lambda [1/yr] on the balanced path, where
    capital and energy constraints both bind.

    Q = K/tau = P/e with P = p' C. Machinery gets u Q, so K = u Q / lambda
    and u = lambda tau. Collectors arrive after delay D, so
    C = (1 - u) Q exp(-lambda D) / lambda. Substituting into Q = p' C / e:
        lambda t_E = (1 - lambda tau) exp(-lambda D),   t_E = e / p'.
    The left side rises and the right side falls on (0, 1/tau): one root.
    This is the Euler-Lotka equation with a maturation delay D.
    Returns 0 if collectors cannot work at r.
    """
    t_E = energy_payback_time(r, p)
    if not np.isfinite(t_E):
        return 0.0
    D = transit_time(r, p)
    g = lambda lam: lam * t_E - (1 - lam * p.tau) * np.exp(-lam * D)
    return brentq(g, 0.0, 1.0 / p.tau)


def analytic_time_to_target(r, f, p):
    """
    Time to reach captured fraction f under the capital-limited bang-bang
    policy, ignoring energy limits and wear [yr]. Returns (T, K_switch).

    Grow K = K0 exp(t/tau) until t_s, then make collectors at rate K_s/tau:
        T = t_s + tau M / K_s + D.
    Minimising over t_s gives K_s = M: the final phase lasts exactly tau.
    Feedstock caps mass: K_s - K0 + M <= y M_belt. When that binds,
    K_s = y M_belt + K0 - M and the final phase is longer.
    """
    M = target_mass(r, f, p)
    K_cap = p.yield_frac * p.M_belt + p.K0 - M
    if K_cap <= p.K0:
        return np.inf, np.nan
    K_s = min(M, K_cap)
    K_s = max(K_s, p.K0)
    T = p.tau * np.log(K_s / p.K0) + p.tau * M / K_s + transit_time(r, p)
    return T, K_s


# ============================================================
# ODE simulation
# ============================================================
def _rhs(t, y, r, u, p, sp, e, D):
    n = p.n_transit
    # Clamp round-off negatives: the stiff transit chain can otherwise be
    # driven through denormals to overflow once production stops.
    K = max(y[0], 0.0)
    T = np.maximum(y[1:1 + n], 0.0)
    C = max(y[1 + n], 0.0)
    X = y[2 + n]

    P = p.P_seed + sp * C                       # W
    Q_cap = K / p.tau                           # kg/yr
    Q_en = P * YR / e                           # kg/yr
    F_eps = 1e-6 * p.M_belt
    avail = float(np.clip((p.M_belt - X) / F_eps, 0.0, 1.0))  # exhausted -> 0
    Q = min(Q_cap, Q_en) * avail

    wear_K = K / p.life_K
    wear_C = C / p.life_C
    k = n / D

    dy = np.empty_like(y)
    dy[0] = u * Q - wear_K
    dy[1] = (1 - u) * Q - k * T[0]
    for j in range(1, n):
        dy[1 + j] = k * (T[j - 1] - T[j])
    dy[1 + n] = k * T[-1] - wear_C
    dy[2 + n] = Q / p.yield_frac
    dy[3 + n] = wear_K + wear_C + Q * (1 / p.yield_frac - 1)
    return dy


def simulate(r, f_target, t_switch, p, u_pre=None, t_max=500.0, dense=False):
    """
    Integrate the throughput ODE for collectors at radius r [AU].

    Policy: allocate u_pre of output to machinery until t_switch, then
    u = 0 (all output becomes collectors). u_pre defaults to the balanced
    value lambda tau. Stops when the captured fraction reaches f_target.

    Returns a dict with 't_hit' (np.inf if never reached) and, if dense,
    the trajectory arrays.
    """
    sp = specific_power(r, p)
    e = energy_per_kg(r, p)
    D = transit_time(r, p)
    if u_pre is None:
        u_pre = balanced_growth_rate(r, p) * p.tau
    M_target = target_mass(r, f_target, p)

    n = p.n_transit
    y0 = np.zeros(4 + n)
    y0[0] = p.K0

    def hit(t, y, *args):
        return y[1 + n] - M_target
    hit.terminal = True
    hit.direction = 1

    ts, ys = [], []
    t_hit = np.inf
    t0 = 0.0
    for u, t_end in ((u_pre, min(t_switch, t_max)), (0.0, t_max)):
        if t_end <= t0:
            continue
        sol = solve_ivp(_rhs, (t0, t_end), y0, args=(r, u, p, sp, e, D),
                        method='DOP853', rtol=1e-8, atol=1e-3, events=hit,
                        dense_output=False, max_step=max(D, 0.05))
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise RuntimeError(f"integration failed at r={r}, t_switch={t_switch}: {sol.message}")
        ts.append(sol.t)
        ys.append(sol.y)
        if sol.t_events[0].size:
            t_hit = float(sol.t_events[0][0])
            break
        y0 = sol.y[:, -1]
        t0 = t_end

    out = {'t_hit': t_hit, 'r': r, 'f_target': f_target,
           't_switch': t_switch, 'u_pre': u_pre}
    if dense:
        t = np.concatenate(ts)
        y = np.concatenate(ys, axis=1)
        out.update(t=t, K=y[0], transit=y[1:1 + n].sum(axis=0), C=y[1 + n],
                   mined=y[2 + n], F=p.M_belt - y[2 + n], W=y[3 + n],
                   f=y[1 + n] / target_mass(r, 1.0, p),
                   P=p.P_seed + sp * y[1 + n])
    return out


def optimize_switch(r, f_target, p, n_grid=20):
    """
    Find the switch time minimising time to reach f_target.
    Returns the simulate() result at the optimum (t_hit = inf if infeasible).
    """
    T_a, _ = analytic_time_to_target(r, f_target, p)
    if not np.isfinite(T_a) or balanced_growth_rate(r, p) <= 0:
        return {'t_hit': np.inf, 'r': r, 'f_target': f_target,
                't_switch': np.nan, 'u_pre': np.nan}
    hi = 1.5 * T_a + 5 * p.tau
    t_max = 3 * hi
    grid = np.linspace(0.0, hi, n_grid)
    vals = [simulate(r, f_target, ts, p, t_max=t_max)['t_hit'] for ts in grid]
    i = int(np.argmin(vals))
    if not np.isfinite(vals[i]):
        return {'t_hit': np.inf, 'r': r, 'f_target': f_target,
                't_switch': np.nan, 'u_pre': np.nan}
    lo_b, hi_b = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    # Large finite penalty keeps the bounded search free of inf arithmetic
    obj = lambda ts: min(simulate(r, f_target, ts, p, t_max=t_max)['t_hit'], 10 * t_max)
    res = minimize_scalar(obj, bounds=(lo_b, hi_b), method='bounded',
                          options={'xatol': 1e-3 * p.tau})
    best = res.x if res.fun <= vals[i] else grid[i]
    return simulate(r, f_target, best, p, t_max=t_max)


# ============================================================
# Figures
# ============================================================
# Categorical slots 1-4 of the validated reference palette, in fixed order.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_trajectory(p, r, f_target, fname):
    import matplotlib.pyplot as plt
    opt = optimize_switch(r, f_target, p)
    sim = simulate(r, f_target, opt['t_switch'], p, t_max=opt['t_hit'] + 1e-9, dense=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 2]})
    for ax in (ax1, ax2):
        _style(ax)
    for key, label, col in (('K', 'Machinery', SERIES[0]),
                            ('transit', 'In transit', SERIES[1]),
                            ('C', 'Collectors', SERIES[2])):
        ax1.semilogy(sim['t'], np.maximum(sim[key], 1.0), color=col, lw=2, label=label)
    ax1.set_ylabel('Mass (kg)')
    ax1.set_ylim(1e5, None)
    ax1.legend(frameon=False, fontsize=9, loc='upper left')
    ax1.set_title(f'Growth path to f = {f_target:g} at r = {r:g} AU '
                  f'(tau = {p.tau:g} yr, sigma = {p.sigma:g} kg/m$^2$)',
                  fontsize=11, color=INK, loc='left')

    ax2.semilogy(sim['t'], np.maximum(sim['f'], 1e-20), color=SERIES[0], lw=2)
    ax2.set_ylim(1e-12, 3 * f_target)
    ax2.axhline(f_target, color=INK_2, lw=1, ls='--')
    ax2.set_ylabel('Captured fraction of L$_\\odot$')
    ax2.set_xlabel('Time (yr)')
    for ax in (ax1, ax2):
        ax.axvline(opt['t_switch'], color=INK_2, lw=1, ls=':')
    ax1.text(opt['t_switch'], ax1.get_ylim()[1], ' switch', fontsize=8,
             color=INK_2, va='top')
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return opt


def plot_time_vs_radius(p, f_target, taus, fname, radii=None):
    import matplotlib.pyplot as plt
    if radii is None:
        radii = np.linspace(0.3, 3.0, 19)
    # Building at the belt needs no transfer; show it as its own point rather
    # than as a spike in a curve where every other radius pays a Hohmann time.
    radii = np.array([r for r in radii if not np.isclose(r, p.a_belt)])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style(ax)
    rows = []
    for tau, col in zip(taus, SERIES):
        pt = replace(p, tau=tau)
        sim_T = np.array([optimize_switch(r, f_target, pt)['t_hit'] for r in radii])
        ana_T = np.array([analytic_time_to_target(r, f_target, pt)[0] for r in radii])
        ax.plot(radii, sim_T, color=col, lw=2, label=f'tau = {tau:g} yr (ODE)')
        ax.plot(radii, ana_T, color=col, lw=1, ls='--')
        ok = np.isfinite(sim_T)
        if ok.any():
            j = int(np.nanargmin(np.where(ok, sim_T, np.nan)))
            ax.plot(radii[j], sim_T[j], 'o', ms=8, color=col,
                    markeredgecolor='white', markeredgewidth=2)
            ax.annotate(f'{sim_T[j]:.0f} yr at {radii[j]:.2f} AU', (radii[j], sim_T[j]),
                        xytext=(6, -12), textcoords='offset points', fontsize=8, color=INK_2)
        T_insitu = optimize_switch(p.a_belt, f_target, pt)['t_hit']
        ax.plot(p.a_belt, T_insitu, 's', ms=8, color=col,
                markeredgecolor='white', markeredgewidth=2)
        rows.append((tau, radii, sim_T, ana_T, T_insitu))
    ax.plot([], [], color=INK_2, lw=1, ls='--', label='analytic bang-bang')
    ax.plot([], [], 's', ms=8, color=INK_2, label='in situ at the belt (no transfer)')
    ax.set_xlabel('Collector radius r (AU)')
    ax.set_ylabel(f'Time to capture f = {f_target:g} (yr)')
    ax.set_title(f'Time to reach f = {f_target:g} vs collector radius '
                 f'(sigma = {p.sigma:g} kg/m$^2$)', fontsize=11, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return rows


def plot_feedstock_frontier(p, sigmas, fname):
    import matplotlib.pyplot as plt
    radii = np.linspace(0.1, 3.0, 200)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style(ax)
    for s, col in zip(sigmas, SERIES):
        ps = replace(p, sigma=s)
        fmax = np.array([feedstock_limit(r, ps) for r in radii])
        ax.loglog(radii, fmax, color=col, lw=2, label=f'sigma = {s:g} kg/m$^2$')
    ax.axhline(1.0, color=INK_2, lw=1, ls='--')
    ax.text(2.0, 1.1, 'full capture', fontsize=8, color=INK_2)
    r_hot = brentq(lambda r: efficiency(r, p) - 1e-9, 0.05, 3.0)
    ax.axvspan(radii[0], r_hot, color=GRID, alpha=0.6, lw=0)
    ax.text(0.11, ax.get_ylim()[0] * 2, f'collectors too hot\n(eta = 0 inside {r_hot:.2f} AU)',
            fontsize=8, color=INK_2)
    ax.set_xlabel('Collector radius r (AU)')
    ax.set_ylabel('Max captured fraction from belt feedstock')
    ax.set_title(f'Feedstock frontier (M_belt = {p.M_belt:.1e} kg, yield = {p.yield_frac:g})',
                 fontsize=11, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def summary_table(p, radii=(0.5, 1.0, 2.0, 2.7)):
    print(f"{'r':>5} {'T_eq':>6} {'eta':>5} {'dv':>6} {'D':>5} {'t_E':>9} "
          f"{'lambda':>7} {'M(f=1)':>9} {'f_max':>7}")
    print(f"{'AU':>5} {'K':>6} {'':>5} {'km/s':>6} {'yr':>5} {'s':>9} "
          f"{'1/yr':>7} {'kg':>9} {'':>7}")
    for r in radii:
        print(f"{r:5.2f} {collector_temperature(r):6.0f} {efficiency(r, p):5.3f} "
              f"{hohmann_dv(p.a_belt, r) / 1e3:6.1f} {transit_time(r, p):5.2f} "
              f"{energy_payback_time(r, p) * YR:9.3g} {balanced_growth_rate(r, p):7.4f} "
              f"{target_mass(r, 1.0, p):9.2e} {feedstock_limit(r, p):7.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--f', type=float, default=0.1, help='target captured fraction')
    ap.add_argument('--r', type=float, default=1.0, help='radius for trajectory plot [AU]')
    ap.add_argument('--tau', type=float, default=1.0, help='machinery replication time [yr]')
    ap.add_argument('--sigma', type=float, default=1e-3, help='collector areal density [kg/m^2]')
    ap.add_argument('--prefix', default='growth', help='output file prefix')
    args = ap.parse_args()

    p = Params(tau=args.tau, sigma=args.sigma)
    print("Collector physics and balanced growth:")
    summary_table(p)
    print()

    opt = plot_trajectory(p, args.r, args.f, f'{args.prefix}_trajectory.png')
    T_a, K_s = analytic_time_to_target(args.r, args.f, p)
    print(f"r = {args.r} AU, f = {args.f}: ODE optimum T = {opt['t_hit']:.2f} yr "
          f"(switch at {opt['t_switch']:.2f} yr); analytic T = {T_a:.2f} yr")

    rows = plot_time_vs_radius(p, args.f, (0.25, 1.0, 4.0), f'{args.prefix}_time_vs_radius.png')
    for tau, radii, sim_T, _, T_insitu in rows:
        j = int(np.nanargmin(np.where(np.isfinite(sim_T), sim_T, np.nan)))
        print(f"  tau = {tau:5.2f} yr: fastest transferred r = {radii[j]:.2f} AU, "
              f"T = {sim_T[j]:.1f} yr; in situ at belt T = {T_insitu:.1f} yr "
              f"(f_max there {feedstock_limit(p.a_belt, p):.2f})")

    plot_feedstock_frontier(p, (1e-3, 1e-2, 1e-1), f'{args.prefix}_feedstock_frontier.png')
    print(f"Saved {args.prefix}_trajectory.png, {args.prefix}_time_vs_radius.png, "
          f"{args.prefix}_feedstock_frontier.png")


if __name__ == '__main__':
    main()
