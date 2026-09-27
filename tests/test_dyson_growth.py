"""Checks of the throughput model against closed-form results."""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import dyson_growth as dg  # noqa: E402

P = dg.Params()


def test_hohmann_earth_mars():
    assert dg.hohmann_dv(1.0, 1.524) == pytest.approx(5.59e3, rel=0.01)
    p = replace(P, a_belt=1.524)
    assert dg.transit_time(1.0, p) * 365.25 == pytest.approx(259, rel=0.01)


def test_collector_temperature_1au():
    assert dg.collector_temperature(1.0) == pytest.approx(331, abs=1)


def test_balanced_rate_series_law():
    """
    D -> 0: 1/lambda = tau + t_E exactly. Small D: to first order the delay
    also adds in series, 1/lambda ~ tau + t_E + D.
    """
    p = replace(P, sigma=10.0, D_min=1e-9)          # t_E ~ 1 yr, comparable to tau
    r = p.a_belt
    t_E = dg.energy_payback_time(r, p)
    assert 0.1 < t_E < 10
    assert dg.balanced_growth_rate(r, p) == pytest.approx(1 / (p.tau + t_E), rel=1e-6)
    p = replace(p, D_min=0.05)
    # second-order term is lambda^2 tau D ~ 1%
    assert dg.balanced_growth_rate(r, p) == pytest.approx(1 / (p.tau + t_E + 0.05), rel=2e-2)


@pytest.mark.parametrize('r', [1.0, 2.0])
def test_simulated_growth_matches_euler_lotka(r):
    """On the balanced allocation the ODE grows at the Euler-Lotka rate."""
    p = replace(P, sigma=10.0, n_transit=30)
    lam = dg.balanced_growth_rate(r, p)
    sim = dg.simulate(r, 1e9, np.inf, p, t_max=60.0, dense=True)
    t, K = sim['t'], sim['K']
    sel = (t > 30) & (t < 60)
    slope = np.polyfit(t[sel], np.log(K[sel]), 1)[0]
    assert slope == pytest.approx(lam, rel=0.02)


def test_mass_conservation():
    p = replace(P, life_K=20.0, life_C=30.0)
    sim = dg.simulate(1.0, 0.05, 25.0, p, dense=True)
    mined = sim['mined']
    total = sim['K'] + sim['transit'] + sim['C'] + sim['W']
    assert np.allclose(mined + p.K0, total, rtol=1e-6)


def test_bang_bang_final_phase_lasts_tau():
    """Capital-limited, near-zero delay: optimum spends ~tau on collectors."""
    r = P.a_belt
    opt = dg.optimize_switch(r, 0.05, P)
    T_a, _ = dg.analytic_time_to_target(r, 0.05, P)
    assert opt['t_hit'] == pytest.approx(T_a, rel=0.02)
    assert opt['t_hit'] - opt['t_switch'] == pytest.approx(P.tau, rel=0.1)


def test_infeasible_beyond_feedstock():
    r = 2.0
    f_too_big = 1.1 * dg.feedstock_limit(r, P)
    assert dg.analytic_time_to_target(r, f_too_big, P)[0] == np.inf
    assert dg.optimize_switch(r, f_too_big, P)['t_hit'] == np.inf


def test_too_hot_collectors_do_not_grow():
    assert dg.efficiency(0.2, P) == 0.0
    assert dg.balanced_growth_rate(0.2, P) == 0.0
