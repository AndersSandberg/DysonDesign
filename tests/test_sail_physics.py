"""
Regression tests for the test particle setup and the solar sail force.

Run with: python -m pytest tests/
"""
import os
import sys

import numpy as np
import pytest
import rebound

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import solar_system_stability_map as ssm  # noqa: E402


def _helio_radius(sim):
    tp, sun = ssm.get_test_particle(sim), sim.particles[0]
    return np.sqrt((tp.x - sun.x) ** 2 + (tp.y - sun.y) ** 2 + (tp.z - sun.z) ** 2)


@pytest.mark.parametrize('a', [0.1, 0.5, 1.0, 2.0])
def test_circular_orbit_is_heliocentric(a):
    """A nominally circular test particle must be circular about the Sun."""
    sim = ssm.setup_test_particle_sim(a, 5.0, 0.0, 0.0)
    orbit = ssm.get_test_particle(sim).orbit(primary=sim.particles[0])
    assert orbit.e < 1e-6
    assert orbit.a == pytest.approx(a, rel=1e-6)


@pytest.mark.parametrize('beta', [0.3, 0.6, 0.9, 0.99])
def test_sail_force_is_applied(beta):
    """
    With the planets made massless, a circular sail orbit must keep its
    radius to within the timestep's splitting error. If the force were
    silently dropped, the particle would follow an ellipse with e = beta.
    """
    sim = ssm.setup_test_particle_sim(1.5, 5.0, 0.0, beta)
    for i in range(1, len(ssm.PLANET_ELEMENTS) + 1):
        sim.particles[i].m = 0.0
    radii = []
    for t in np.linspace(0.5, 30.0, 120):
        sim.integrate(t, exact_finish_time=0)
        radii.append(_helio_radius(sim))
    ssm.check_force_status()
    assert ssm._FORCE_CALLS > 0
    assert (max(radii) - min(radii)) / 1.5 < 3e-3


def test_force_error_is_reported():
    """Errors inside the ctypes callback must surface, not vanish."""
    ssm.setup_test_particle_sim(1.0, 0.0, 0.0, 0.3)  # resets the force state
    ssm._FORCE_ERROR = AttributeError('simulated failure')
    with pytest.raises(RuntimeError, match='simulated failure'):
        ssm.check_force_status()


def _two_body_sail(beta, dx=0.0):
    ssm._LIGHTNESS = beta
    ssm._FORCE_CALLS = 0
    ssm._FORCE_ERROR = None
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.add(m=1.0)
    sim.add(m=0.0, a=1.0, e=0.1, inc=0.2, primary=sim.particles[0])
    tp = sim.particles[1]
    s = (1.0 - beta) ** 0.5
    tp.vx *= s
    tp.vy *= s
    tp.vz *= s
    tp.x += dx
    sim.move_to_com()
    sim.integrator = 'whfast'
    sim.dt = 0.002
    sim.additional_forces = ssm._radiation_force
    sim.force_is_velocity_dependent = 0
    return sim


def test_variational_force_matches_finite_difference():
    """
    The tangent vector propagated with the Jacobian of the radiation force
    must match a central finite difference of two real trajectories.
    """
    beta, T, eps = 0.5, 3.0, 1e-7

    sim = _two_body_sail(beta)
    var = sim.add_variation()
    var.particles[1].x = 1.0
    sim.integrate(T, exact_finish_time=0)
    ssm.check_force_status()
    d_var = np.array(var.particles[1].xyz) - np.array(var.particles[0].xyz)

    ends = []
    for sign in (+1, -1):
        s = _two_body_sail(beta, dx=sign * eps)
        s.integrate(T, exact_finish_time=0)
        ends.append(np.array(s.particles[1].xyz) - np.array(s.particles[0].xyz))
    d_fd = (ends[0] - ends[1]) / (2 * eps)

    assert np.allclose(d_var, d_fd, rtol=1e-4, atol=1e-6 * np.linalg.norm(d_fd))


def test_resonances_are_reduced_ratios():
    from math import gcd
    for res in ssm.get_resonance_locations(0.1, 3.0):
        assert gcd(res['p'], res['q']) == 1


@pytest.mark.parametrize('beta', [0.0, 0.5, 0.9])
def test_resonance_shift_with_lightness(beta):
    """Resonance semimajor axes scale as (1-beta)^(1/3)."""
    base = {(r['planet'], r['ratio']): r['a'] for r in ssm.get_resonance_locations(0.01, 100.0)}
    for r in ssm.get_resonance_locations(0.01, 100.0, beta):
        assert r['a'] == pytest.approx(base[(r['planet'], r['ratio'])] * (1 - beta) ** (1 / 3))


def test_timestep_bounds_splitting_error():
    for a in (0.1, 1.0, 3.0):
        for beta in (0.1, 0.5, 0.9, 0.99):
            dt = ssm.choose_timestep(a, 0.0, beta)
            n = 2 * np.pi / a ** 1.5
            assert beta / (1 - beta) * (n * dt) ** 2 <= ssm.SPLIT_TOL * (1 + 1e-12)
    with pytest.raises(ValueError):
        ssm.choose_timestep(1.0, 0.0, 1.0)
