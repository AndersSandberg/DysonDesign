#!/usr/bin/env python3
"""
Solar System Orbital Stability Map using MEGNO
Computes stability of test particle orbits across (a, i) space.

Features:
  - Parallel computation using multiprocessing
  - Row-based checkpointing (saves after every N rows)
  - Auto-resume from partial results

Requires: pip install rebound numpy matplotlib tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rebound
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
import os
import time
import json
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - Edit these parameters
# ============================================================
CONFIG = {
    'a_range': (0.1, 3.0),          # Semimajor axis range in AU
    'i_range': (0, 180),             # Inclination range in degrees
    'n_a': 20,                      # Grid points in semimajor axis
    'n_i': 10,                      # Grid points in inclination
    'e_test': 0.0,                   # Test particle eccentricity
    'lightness': 0.0,                # Solar sail lightness beta (0=no sail, 1=hovering)
    'integration_time': 5000.0,      # Integration time in years
    'output_prefix': 'stability',    # Prefix for output files
    'n_workers': 0,                  # 0 = auto (use all cores)
    'checkpoint_every': 1,           # Save checkpoint every N rows
}

# ============================================================
# J2000 Orbital Elements (hardcoded - no network needed)
# Source: JPL Horizons, epoch J2000.0 (JD 2451545.0)
# Format: (a [AU], e, inc [deg], Omega [deg], omega [deg], M [deg], mass [Msun])
# ============================================================
PLANET_ELEMENTS = {
    'Mercury': (0.38710, 0.20563, 7.005, 48.331, 29.124, 174.796, 1.6601e-7),
    'Venus':   (0.72333, 0.00677, 3.395, 76.680, 54.884, 50.416,  2.4478e-6),
    'Earth':   (1.00000, 0.01671, 0.000, -11.261, 102.937, 357.517, 3.0027e-6),
    'Mars':    (1.52368, 0.09340, 1.850, 49.558, 286.502, 19.373,  3.2272e-7),
    'Jupiter': (5.20260, 0.04849, 1.303, 100.464, 273.867, 20.020, 9.5479e-4),
    'Saturn':  (9.55491, 0.05551, 2.489, 113.666, 339.392, 317.020, 2.8588e-4),
    'Uranus':  (19.2184, 0.04630, 0.773, 74.006, 96.998, 142.238, 4.3662e-5),
    'Neptune': (30.1104, 0.00899, 1.770, 131.784, 276.336, 256.228, 5.1514e-5),
}

# Planet semimajor axes for resonance calculations
PLANET_A = {name: elem[0] for name, elem in PLANET_ELEMENTS.items()}

# ============================================================
# Mean-motion resonance locations
# ============================================================
def get_resonance_locations(a_min, a_max, lightness=0.0):
    """
    Calculate mean-motion resonance locations for all planets.
    
    For a solar sail with lightness beta, the test particle's mean motion is
    n = sqrt(GM(1-beta)/a^3), so resonance locations shift:
      a_res = a_planet * (q/p)^(2/3) * (1-beta)^(1/3)
    
    Returns list of dicts: {a, planet, ratio_str, order}
    where order = |p - q| (lower order = stronger resonance).
    """
    resonances = []
    label_planets = ['Venus', 'Earth', 'Mars', 'Jupiter']
    beta_shift = (1.0 - lightness) ** (1.0 / 3.0) if lightness < 1.0 else 0.0

    for planet_name in label_planets:
        a_p = PLANET_A[planet_name]
        for p in range(1, 8):
            for q in range(1, 8):
                if p == q:
                    continue
                a_res = a_p * (q / p) ** (2.0 / 3.0) * beta_shift
                if a_min <= a_res <= a_max:
                    order = abs(p - q)
                    resonances.append({
                        'a': a_res,
                        'planet': planet_name,
                        'ratio': f'{p}:{q}',
                        'order': order,
                        'p': p, 'q': q,
                    })
    return resonances


# ============================================================
# Solar System simulation setup
# ============================================================
def create_solar_system_sim():
    """Create a REBOUND simulation with the Sun + 8 planets using hardcoded elements."""
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')

    sim.add(m=1.0)  # Sun

    for name, (a, e, inc, Omega, omega, M, mass) in PLANET_ELEMENTS.items():
        sim.add(
            m=mass, a=a, e=e,
            inc=np.radians(inc),
            Omega=np.radians(Omega),
            omega=np.radians(omega),
            M=np.radians(M),
        )

    sim.move_to_com()
    return sim


# ============================================================
# Solar sail radiation pressure force
# ============================================================
# Module-level variable for the force callback (safe with multiprocessing:
# each worker process gets its own copy of module globals)
_LIGHTNESS = 0.0

def _radiation_force(reb_sim):
    """
    Additional radial force for a solar sail with lightness beta.
    F_rad = beta * G * M_sun / r^2 (outward), applied to test particle only.
    
    Also applies the variational (Jacobian) force to the corresponding
    shadow particle, which is required for correct MEGNO computation.
    
    The Jacobian of F_i = C * r_i / r^3 is:
        dF_i/dr_j = C * (delta_ij / r^3  -  3 * r_i * r_j / r^5)
    
    Applied to the shadow displacement delta_r:
        delta_a_i = C * (delta_r_i / r^3  -  3 * r_i * (r . delta_r) / r^5)
    """
    if _LIGHTNESS == 0.0:
        return
    sim = reb_sim.contents
    ps = sim.particles
    sun = ps[0]
    n_real = sim.N_real
    tp_idx = n_real - 1  # test particle is last real particle
    tp = ps[tp_idx]
    
    dx = tp.x - sun.x
    dy = tp.y - sun.y
    dz = tp.z - sun.z
    r2 = dx * dx + dy * dy + dz * dz
    r3 = r2 * r2 ** 0.5
    
    C = _LIGHTNESS * sim.G * sun.m
    
    # --- Force on real test particle ---
    f = C / r3
    tp.ax += f * dx
    tp.ay += f * dy
    tp.az += f * dz
    
    # --- Variational force on shadow particle (needed for MEGNO) ---
    if sim.N > n_real:
        r5 = r3 * r2
        shadow_tp = ps[tp_idx + n_real]
        shadow_sun = ps[n_real]  # shadow particle for Sun
        
        # Deviation of relative position (test particle - Sun)
        ddx = shadow_tp.x - shadow_sun.x
        ddy = shadow_tp.y - shadow_sun.y
        ddz = shadow_tp.z - shadow_sun.z
        
        # r . delta_r
        rddr = dx * ddx + dy * ddy + dz * ddz
        
        # Jacobian applied to deviation
        shadow_tp.ax += C * (ddx / r3 - 3.0 * dx * rddr / r5)
        shadow_tp.ay += C * (ddy / r3 - 3.0 * dy * rddr / r5)
        shadow_tp.az += C * (ddz / r3 - 3.0 * dz * rddr / r5)


# ============================================================
# Single MEGNO computation (must be top-level for pickling)
# ============================================================
def compute_megno_single(args):
    """
    Compute MEGNO for a single test particle at (a, inc).
    Takes a tuple (a, inc_deg, e_test, lightness, integration_time) for multiprocessing.
    Returns MEGNO value or NaN on failure.
    """
    a, inc_deg, e_test, lightness, integration_time = args
    
    # Set module-level lightness for the force callback
    global _LIGHTNESS
    _LIGHTNESS = lightness
    
    try:
        sim = create_solar_system_sim()

        sim.integrator = "whfast"
        sim.dt = min(a ** 1.5, 0.5) * 0.05

        # Add test particle BEFORE init_megno
        sim.add(
            m=0.0, a=a, e=e_test,
            inc=np.radians(inc_deg),
            Omega=0.0, omega=0.0, M=0.0,
        )
        
        # Adjust velocity for solar sail: circular orbit in effective potential
        # needs v_circ = v_kepler * sqrt(1 - beta)
        if lightness > 0.0 and lightness < 1.0:
            tp = sim.particles[sim.N - 1]  # test particle (last added)
            scale = (1.0 - lightness) ** 0.5
            tp.vx *= scale
            tp.vy *= scale
            tp.vz *= scale

        sim.move_to_com()
        
        # Enable radiation pressure force
        if lightness > 0.0:
            sim.additional_forces = _radiation_force
            sim.force_is_velocity_dependent = 0
        
        sim.init_megno()

        # Integrate
        n_steps = max(100, int(integration_time / (a ** 1.5 * 0.5)))
        n_steps = min(n_steps, 5000)

        times = np.linspace(0, integration_time, n_steps)
        for t in times[1:]:
            sim.integrate(t)

        megno = sim.megno()
        if np.isfinite(megno) and megno > 0:
            return megno
        else:
            return np.nan

    except Exception:
        return np.nan


def compute_row(args):
    """
    Compute one full row (one inclination, all semimajor axes).
    Returns (row_index, array of MEGNO values).
    """
    row_idx, inc_deg, a_values, e_test, lightness, integration_time = args
    results = np.empty(len(a_values))
    for i, a in enumerate(a_values):
        results[i] = compute_megno_single((a, inc_deg, e_test, lightness, integration_time))
    return row_idx, results


# ============================================================
# Checkpointing
# ============================================================
def get_checkpoint_path(config):
    return f"{config['output_prefix']}_checkpoint.npz"

def get_checkpoint_meta_path(config):
    return f"{config['output_prefix']}_checkpoint_meta.json"

def save_checkpoint(a_values, i_values, megno_map, completed_rows, config):
    """Save current state to checkpoint files."""
    np.savez(
        get_checkpoint_path(config),
        a_values=a_values,
        i_values=i_values,
        megno_map=megno_map,
    )
    meta = {
        'completed_rows': sorted(completed_rows),
        'config': {k: v if not isinstance(v, tuple) else list(v) for k, v in config.items()},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(get_checkpoint_meta_path(config), 'w') as f:
        json.dump(meta, f, indent=2)

def load_checkpoint(config):
    """
    Load checkpoint if it exists and config matches.
    Returns (megno_map, completed_rows) or (None, set()) if no valid checkpoint.
    """
    ckpt_path = get_checkpoint_path(config)
    meta_path = get_checkpoint_meta_path(config)

    if not (os.path.exists(ckpt_path) and os.path.exists(meta_path)):
        return None, set()

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        # Verify config compatibility
        saved_cfg = meta['config']
        for key in ['a_range', 'i_range', 'n_a', 'n_i', 'e_test', 'lightness', 'integration_time']:
            saved_val = saved_cfg.get(key)
            current_val = config[key]
            if isinstance(current_val, tuple):
                saved_val = tuple(saved_val)
            if saved_val != current_val:
                print(f"  Checkpoint config mismatch on '{key}': {saved_val} vs {current_val}")
                return None, set()

        data = np.load(ckpt_path)
        megno_map = data['megno_map']
        completed_rows = set(meta['completed_rows'])

        print(f"  Loaded checkpoint: {len(completed_rows)}/{config['n_i']} rows completed")
        print(f"  Last saved: {meta['timestamp']}")
        return megno_map, completed_rows

    except Exception as ex:
        print(f"  Failed to load checkpoint: {ex}")
        return None, set()


# ============================================================
# Parallel grid computation with checkpointing
# ============================================================
def compute_stability_map(config):
    """Compute MEGNO over the (a, i) grid with parallelism and checkpointing."""
    a_values = np.linspace(config['a_range'][0], config['a_range'][1], config['n_a'])
    i_values = np.linspace(config['i_range'][0], config['i_range'][1], config['n_i'])

    n_workers = config['n_workers'] if config['n_workers'] > 0 else cpu_count()
    checkpoint_every = config['checkpoint_every']

    # Try to resume from checkpoint
    print("Checking for checkpoint...")
    megno_map, completed_rows = load_checkpoint(config)
    if megno_map is None:
        megno_map = np.full((config['n_i'], config['n_a']), np.nan)
        completed_rows = set()
        print("  Starting fresh.")

    # Build list of remaining rows
    remaining = [(j, i_values[j], a_values, config['e_test'], config['lightness'],
                  config['integration_time'])
                 for j in range(config['n_i']) if j not in completed_rows]

    if not remaining:
        print("All rows already computed!")
        return a_values, i_values, megno_map

    total_pts = len(remaining) * config['n_a']
    done_pts = len(completed_rows) * config['n_a']
    all_pts = config['n_a'] * config['n_i']

    print(f"\nComputing {len(remaining)} remaining rows ({total_pts} points) using {n_workers} workers")
    print(f"  Already done: {len(completed_rows)}/{config['n_i']} rows ({done_pts}/{all_pts} points)")
    print(f"  Integration time: {config['integration_time']} yr")
    print(f"  Checkpoint every {checkpoint_every} row(s)")
    print()

    t_start = time.time()
    rows_since_checkpoint = 0

    # Use imap_unordered for best throughput + live progress
    with Pool(processes=n_workers) as pool:
        with tqdm(total=len(remaining), desc="Rows", unit="row") as pbar:
            for row_idx, row_data in pool.imap_unordered(compute_row, remaining):
                megno_map[row_idx, :] = row_data
                completed_rows.add(row_idx)
                rows_since_checkpoint += 1
                pbar.update(1)

                # Update ETA
                elapsed = time.time() - t_start
                rows_done = pbar.n
                if rows_done > 0:
                    rate = elapsed / rows_done
                    remaining_time = rate * (len(remaining) - rows_done)
                    pbar.set_postfix_str(
                        f"~{remaining_time/60:.0f}min left, "
                        f"{len(completed_rows)}/{config['n_i']} total"
                    )

                # Checkpoint
                if rows_since_checkpoint >= checkpoint_every:
                    save_checkpoint(a_values, i_values, megno_map, completed_rows, config)
                    rows_since_checkpoint = 0

    # Final save
    save_checkpoint(a_values, i_values, megno_map, completed_rows, config)

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  {total_pts / elapsed:.1f} points/sec")

    return a_values, i_values, megno_map


# ============================================================
# Visualization
# ============================================================
def plot_stability_map(a_values, i_values, megno_map, config, save=True):
    """
    Create the MEGNO stability map figure.

    Color scale:
      MEGNO ~ 2   -> regular/quasi-periodic (green)
      MEGNO >> 2  -> chaotic (red/yellow)
      MEGNO < 2   -> possibly not converged yet (blue)
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.cm.RdYlGn_r

    megno_display = np.clip(megno_map, 0.5, 50)

    im = ax.pcolormesh(
        a_values, i_values, megno_display,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=0.5, vmax=50),
        shading='nearest',
        rasterized=True,
    )

    cbar = plt.colorbar(im, ax=ax, label='MEGNO <Y>', pad=0.02)
    cbar.set_ticks([0.5, 1, 2, 5, 10, 20, 50])
    cbar.set_ticklabels(['0.5', '1', '2', '5', '10', '20', '50'])

    # --- Resonance lines ---
    beta = config.get('lightness', 0.0)
    resonances = get_resonance_locations(config['a_range'][0], config['a_range'][1], beta)

    plotted_positions = []

    for res in sorted(resonances, key=lambda r: r['order']):
        if res['order'] > 2:
            continue

        a_res = res['a']
        too_close = any(abs(a_res - ap) < 0.03 for ap in plotted_positions)

        if res['order'] == 1:
            lw, alpha, ls = 1.2, 0.7, '-'
        else:
            lw, alpha, ls = 0.8, 0.5, '--'

        color_map = {
            'Venus': '#1f77b4',
            'Earth': '#2ca02c',
            'Mars': '#d62728',
            'Jupiter': '#ff7f0e',
        }
        color = color_map.get(res['planet'], 'gray')

        ax.axvline(a_res, color=color, linewidth=lw, alpha=alpha, linestyle=ls)

        if not too_close:
            label = f"{res['ratio']} {res['planet'][0]}"
            ax.text(
                a_res, config['i_range'][1] + 3, label,
                ha='center', va='bottom', fontsize=6,
                color=color, rotation=90, clip_on=False,
            )
            plotted_positions.append(a_res)

    # --- Planet locations ---
    for planet_name, planet_a in PLANET_A.items():
        if config['a_range'][0] <= planet_a <= config['a_range'][1]:
            ax.axvline(planet_a, color='white', linewidth=2.5, alpha=0.8)
            ax.axvline(planet_a, color='black', linewidth=1.5, alpha=0.8, linestyle='-')
            ax.text(
                planet_a, config['i_range'][0] - 5,
                planet_name[:3],
                ha='center', va='top', fontsize=9, fontweight='bold',
                clip_on=False,
            )

    # --- Labels and formatting ---
    ax.set_xlabel('Semimajor Axis (AU)', fontsize=13)
    ax.set_ylabel('Inclination (deg)', fontsize=13)
    beta_str = f', beta={beta:.2f}' if beta > 0 else ''
    ax.set_title(
        f'Solar System Orbital Stability (MEGNO, {config["integration_time"]:.0f} yr, e={config["e_test"]}{beta_str})',
        fontsize=14, fontweight='bold',
    )
    ax.set_xlim(config['a_range'])
    ax.set_ylim(config['i_range'])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=1.2, label='Venus MMR'),
        Line2D([0], [0], color='#2ca02c', lw=1.2, label='Earth MMR'),
        Line2D([0], [0], color='#d62728', lw=1.2, label='Mars MMR'),
        Line2D([0], [0], color='#ff7f0e', lw=1.2, label='Jupiter MMR'),
        Line2D([0], [0], color='black', lw=1.5, label='Planet location'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if save:
        prefix = config['output_prefix']
        for ext in ['png', 'pdf']:
            fname = f'{prefix}_megno_map.{ext}'
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"  Saved: {fname}")

    return fig


# ============================================================
# 1D radial profile: MEGNO collapsed across inclinations
# ============================================================
def plot_1d_profile(a_values, i_values, megno_map, config, save=True,
                    i_range_deg=None):
    """
    Plot MEGNO aggregated across inclinations as a function of semimajor axis.
    
    Parameters:
        i_range_deg: optional (min, max) tuple in degrees to restrict which
                     inclinations are included. None = use all rows.
    
    Shows three curves:
      - Mean MEGNO (overall troublesomeness)
      - Median MEGNO (robust to outliers)
      - Fraction chaotic (MEGNO > 2)
    """
    # Select inclination range
    if i_range_deg is not None:
        mask = (i_values >= i_range_deg[0]) & (i_values <= i_range_deg[1])
        data = megno_map[mask, :]
        i_label = f"i in [{i_range_deg[0]:.0f}, {i_range_deg[1]:.0f}] deg"
    else:
        data = megno_map
        i_label = f"i in [{config['i_range'][0]:.0f}, {config['i_range'][1]:.0f}] deg"

    # Compute column statistics (ignoring NaNs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(data, axis=0)
        col_median = np.nanmedian(data, axis=0)
        col_frac_chaotic = np.nansum(data > 2, axis=0) / np.sum(np.isfinite(data), axis=0)

    # --- Figure with two panels sharing x-axis ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: mean and median MEGNO
    ax1.semilogy(a_values, col_mean, color='#d62728', linewidth=0.8, alpha=0.9,
                 label='Mean MEGNO')
    ax1.semilogy(a_values, col_median, color='#1f77b4', linewidth=0.8, alpha=0.9,
                 label='Median MEGNO')
    ax1.axhline(2.0, color='gray', linewidth=1, linestyle='--', alpha=0.5,
                label='MEGNO = 2 (regular)')
    ax1.set_ylabel('MEGNO', fontsize=12)
    ax1.set_ylim(0.5, 100)
    ax1.legend(fontsize=9, loc='upper left')
    beta = config.get('lightness', 0.0)
    beta_str = f', beta={beta:.2f}' if beta > 0 else ''
    ax1.set_title(
        f'Radial Stability Profile ({i_label}, {config["integration_time"]:.0f} yr{beta_str})',
        fontsize=13, fontweight='bold',
    )

    # Bottom panel: fraction chaotic
    ax2.fill_between(a_values, 0, col_frac_chaotic, color='#d62728', alpha=0.3)
    ax2.plot(a_values, col_frac_chaotic, color='#d62728', linewidth=0.8)
    ax2.set_ylabel('Fraction chaotic', fontsize=12)
    ax2.set_xlabel('Semimajor Axis (AU)', fontsize=12)
    ax2.set_ylim(0, 1.05)

    # --- Add resonance lines and planet markers to both panels ---
    beta = config.get('lightness', 0.0)
    resonances = get_resonance_locations(config['a_range'][0], config['a_range'][1], beta)
    plotted_positions = []

    for res in sorted(resonances, key=lambda r: r['order']):
        if res['order'] > 2:
            continue
        a_res = res['a']
        too_close = any(abs(a_res - ap) < 0.03 for ap in plotted_positions)

        if res['order'] == 1:
            lw, alpha, ls = 0.8, 0.5, '-'
        else:
            lw, alpha, ls = 0.5, 0.3, '--'

        color_map = {
            'Venus': '#1f77b4', 'Earth': '#2ca02c',
            'Mars': '#d62728', 'Jupiter': '#ff7f0e',
        }
        color = color_map.get(res['planet'], 'gray')

        for ax in (ax1, ax2):
            ax.axvline(a_res, color=color, linewidth=lw, alpha=alpha, linestyle=ls)

        if not too_close:
            label = f"{res['ratio']} {res['planet'][0]}"
            ax1.text(a_res, 110, label, ha='center', va='bottom', fontsize=5,
                     color=color, rotation=90, clip_on=False)
            plotted_positions.append(a_res)

    # Planet locations
    for planet_name, planet_a in PLANET_A.items():
        if config['a_range'][0] <= planet_a <= config['a_range'][1]:
            for ax in (ax1, ax2):
                ax.axvline(planet_a, color='black', linewidth=1.2, alpha=0.4, linestyle=':')
            ax2.text(planet_a, -0.08, planet_name[:3], ha='center', va='top',
                     fontsize=8, fontweight='bold', clip_on=False)

    ax2.set_xlim(config['a_range'])
    plt.tight_layout()

    if save:
        prefix = config['output_prefix']
        suffix = ''
        if i_range_deg is not None:
            suffix = f'_i{i_range_deg[0]:.0f}-{i_range_deg[1]:.0f}'
        for ext in ['png', 'pdf']:
            fname = f'{prefix}_1d_profile{suffix}.{ext}'
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"  Saved: {fname}")

    return fig


# ============================================================
# Separate plotting from data
# ============================================================
def plot_from_file(npz_path, config):
    """Load saved data and re-plot (2D map + 1D profile)."""
    data = np.load(npz_path, allow_pickle=True)
    a_values = data['a_values']
    i_values = data['i_values']
    megno_map = data['megno_map']

    fig1 = plot_stability_map(a_values, i_values, megno_map, config, save=True)
    fig2 = plot_1d_profile(a_values, i_values, megno_map, config, save=True)
    plt.show()
    return fig1, fig2


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Solar System Orbital Stability Map using MEGNO")
    print("=" * 70)
    print()
    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    n_workers = CONFIG['n_workers'] if CONFIG['n_workers'] > 0 else cpu_count()
    print(f"  detected cores: {cpu_count()}, will use: {n_workers}")
    print()

    # ---- Quick diagnostic ----
    beta = CONFIG['lightness']
    print(f"Running diagnostic (a=1.5 AU, i=10deg, beta={beta})...")
    try:
        sim = create_solar_system_sim()
        sim.integrator = "whfast"
        sim.dt = 0.05
        print(f"  Particles: {sim.N}, REBOUND {rebound.__version__}")

        sim.add(m=0.0, a=1.5, e=0.0, inc=np.radians(10), Omega=0, omega=0, M=0)
        
        # Adjust velocity for sail
        if beta > 0.0 and beta < 1.0:
            tp = sim.particles[sim.N - 1]
            scale = (1.0 - beta) ** 0.5
            tp.vx *= scale
            tp.vy *= scale
            tp.vz *= scale
            print(f"  Velocity scaled by sqrt(1-beta) = {scale:.4f}")
        
        sim.move_to_com()
        
        if beta > 0.0:
            global _LIGHTNESS
            _LIGHTNESS = beta
            sim.additional_forces = _radiation_force
            sim.force_is_velocity_dependent = 0
            print(f"  Radiation pressure enabled (beta={beta})")
        
        sim.init_megno()

        sim.integrate(100.0)
        print(f"  MEGNO after 100 yr:  {sim.megno():.4f}")
        sim.integrate(1000.0)
        print(f"  MEGNO after 1000 yr: {sim.megno():.4f}")
        print("  Diagnostic PASSED")
    except Exception as ex:
        print(f"  Diagnostic FAILED: {type(ex).__name__}: {ex}")
        print("  Fix this before running the full grid!")
        return
    print()

    # ---- Check for existing results ----
    npz_final = f"{CONFIG['output_prefix']}_data.npz"
    ckpt_path = get_checkpoint_path(CONFIG)

    if os.path.exists(npz_final):
        print(f"Found completed results: {npz_final}")
        print("  (p) Re-plot only")
        print("  (r) Re-compute from scratch")
        print("  (c) Continue from checkpoint (if available)")
        resp = input("Choice [p/r/c]: ").strip().lower()
        if resp == 'p':
            plot_from_file(npz_final, CONFIG)
            return
        elif resp == 'r':
            for f in [ckpt_path, get_checkpoint_meta_path(CONFIG)]:
                if os.path.exists(f):
                    os.remove(f)
    elif os.path.exists(ckpt_path):
        print(f"Found checkpoint: {ckpt_path}")
        print("  (c) Continue from checkpoint")
        print("  (r) Re-compute from scratch")
        resp = input("Choice [c/r]: ").strip().lower()
        if resp == 'r':
            for f in [ckpt_path, get_checkpoint_meta_path(CONFIG)]:
                if os.path.exists(f):
                    os.remove(f)

    # ---- Compute ----
    a_values, i_values, megno_map = compute_stability_map(CONFIG)

    # Save final results
    print("\nSaving final data...")
    np.savez(
        npz_final,
        a_values=a_values,
        i_values=i_values,
        megno_map=megno_map,
        config=CONFIG,
    )
    print(f"  Saved: {npz_final}")

    # Plot 2D map
    print("\nCreating 2D stability map...")
    fig = plot_stability_map(a_values, i_values, megno_map, CONFIG)

    # Plot 1D radial profile
    print("\nCreating 1D radial profile...")
    fig2 = plot_1d_profile(a_values, i_values, megno_map, CONFIG)
    # Optional: also make a low-inclination-only profile
    plot_1d_profile(a_values, i_values, megno_map, CONFIG, i_range_deg=(0, 30))

    # Statistics
    valid = megno_map[np.isfinite(megno_map)]
    if len(valid) > 0:
        print(f"\nStatistics ({len(valid)} valid points of {megno_map.size}):")
        print(f"  Mean MEGNO:  {np.mean(valid):.3f}")
        print(f"  Median MEGNO: {np.median(valid):.3f}")
        print(f"  Chaotic (>2): {np.sum(valid > 2) / len(valid) * 100:.1f}%")
        print(f"  Regular (<=2): {np.sum(valid <= 2) / len(valid) * 100:.1f}%")
    else:
        print("\nNo valid data points!")

    # Clean up checkpoint files after successful completion
    for f in [get_checkpoint_path(CONFIG), get_checkpoint_meta_path(CONFIG)]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Cleaned up: {f}")

    plt.show()


# ################################################################
# (a, beta) MODE - Sweep lightness at fixed inclination
# ################################################################

CONFIG_AB = {
    'a_range': (0.1, 3.0),          # Semimajor axis range in AU
    'beta_range': (0.0, 0.99),       # Lightness range (0 = Keplerian, <1)
    'n_a': 40,                      # Grid points in semimajor axis
    'n_beta': 10,                   # Grid points in lightness
    'inc_deg': 0.0,                  # Fixed inclination in degrees
    'e_test': 0.0,                   # Test particle eccentricity
    'integration_time': 5000.0,      # Integration time in years
    'output_prefix': 'stability_ab', # Prefix for output files
    'n_workers': 0,                  # 0 = auto (use all cores)
    'checkpoint_every': 1,           # Save checkpoint every N rows
}


def compute_row_ab(args):
    """
    Compute one row of the (a, beta) grid: one beta value, all semimajor axes.
    Returns (row_index, array of MEGNO values).
    """
    row_idx, beta_val, a_values, inc_deg, e_test, integration_time = args
    results = np.empty(len(a_values))
    for i, a in enumerate(a_values):
        results[i] = compute_megno_single((a, inc_deg, e_test, beta_val, integration_time))
    return row_idx, results


def compute_ab_map(config):
    """Compute MEGNO over the (a, beta) grid with parallelism and checkpointing."""
    a_values = np.linspace(config['a_range'][0], config['a_range'][1], config['n_a'])
    beta_values = np.linspace(config['beta_range'][0], config['beta_range'][1], config['n_beta'])

    n_workers = config['n_workers'] if config['n_workers'] > 0 else cpu_count()
    checkpoint_every = config['checkpoint_every']

    # Checkpointing - reuse same mechanism with adapted paths
    ckpt_path = f"{config['output_prefix']}_checkpoint.npz"
    meta_path = f"{config['output_prefix']}_checkpoint_meta.json"

    megno_map = None
    completed_rows = set()

    if os.path.exists(ckpt_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            saved_cfg = meta['config']
            match = True
            for key in ['a_range', 'beta_range', 'n_a', 'n_beta', 'inc_deg', 'e_test', 'integration_time']:
                sv = saved_cfg.get(key)
                cv = config[key]
                if isinstance(cv, tuple):
                    sv = tuple(sv)
                if sv != cv:
                    print(f"  Checkpoint config mismatch on '{key}': {sv} vs {cv}")
                    match = False
                    break
            if match:
                data = np.load(ckpt_path)
                megno_map = data['megno_map']
                completed_rows = set(meta['completed_rows'])
                print(f"  Loaded checkpoint: {len(completed_rows)}/{config['n_beta']} rows completed")
        except Exception as ex:
            print(f"  Failed to load checkpoint: {ex}")

    if megno_map is None:
        megno_map = np.full((config['n_beta'], config['n_a']), np.nan)
        completed_rows = set()
        print("  Starting fresh.")

    # Build remaining work
    remaining = [(j, beta_values[j], a_values, config['inc_deg'], config['e_test'],
                  config['integration_time'])
                 for j in range(config['n_beta']) if j not in completed_rows]

    if not remaining:
        print("All rows already computed!")
        return a_values, beta_values, megno_map

    total_pts = len(remaining) * config['n_a']
    print(f"\nComputing {len(remaining)} remaining rows ({total_pts} points) using {n_workers} workers")
    print(f"  Fixed inclination: {config['inc_deg']} deg")
    print(f"  Beta range: {config['beta_range']}")
    print()

    t_start = time.time()
    rows_since_ckpt = 0

    def _save_ckpt():
        np.savez(ckpt_path, a_values=a_values, beta_values=beta_values, megno_map=megno_map)
        m = {'completed_rows': sorted(completed_rows),
             'config': {k: v if not isinstance(v, tuple) else list(v) for k, v in config.items()},
             'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(meta_path, 'w') as f:
            json.dump(m, f, indent=2)

    with Pool(processes=n_workers) as pool:
        with tqdm(total=len(remaining), desc="Rows (beta)", unit="row") as pbar:
            for row_idx, row_data in pool.imap_unordered(compute_row_ab, remaining):
                megno_map[row_idx, :] = row_data
                completed_rows.add(row_idx)
                rows_since_ckpt += 1
                pbar.update(1)

                elapsed = time.time() - t_start
                rows_done = pbar.n
                if rows_done > 0:
                    remaining_time = (elapsed / rows_done) * (len(remaining) - rows_done)
                    pbar.set_postfix_str(
                        f"~{remaining_time/60:.0f}min left, "
                        f"{len(completed_rows)}/{config['n_beta']} total"
                    )

                if rows_since_ckpt >= checkpoint_every:
                    _save_ckpt()
                    rows_since_ckpt = 0

    _save_ckpt()

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  {total_pts / elapsed:.1f} points/sec")

    return a_values, beta_values, megno_map


def plot_ab_map(a_values, beta_values, megno_map, config, save=True):
    """
    Plot MEGNO over (a, beta) space with resonance curves.
    
    Resonance locations depend on beta: a_res(beta) = a_planet * (q/p)^(2/3) * (1-beta)^(1/3)
    These trace curves in (a, beta) space rather than vertical lines.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    megno_display = np.clip(megno_map, 0.5, 50)

    im = ax.pcolormesh(
        a_values, beta_values, megno_display,
        cmap=plt.cm.RdYlGn_r,
        norm=mcolors.LogNorm(vmin=0.5, vmax=50),
        shading='nearest',
        rasterized=True,
    )

    cbar = plt.colorbar(im, ax=ax, label='MEGNO <Y>', pad=0.02)
    cbar.set_ticks([0.5, 1, 2, 5, 10, 20, 50])
    cbar.set_ticklabels(['0.5', '1', '2', '5', '10', '20', '50'])

    # --- Resonance curves: a_res(beta) = a_p * (q/p)^(2/3) * (1-beta)^(1/3) ---
    beta_fine = np.linspace(config['beta_range'][0], config['beta_range'][1], 500)
    label_planets = ['Venus', 'Earth', 'Mars', 'Jupiter']
    color_map = {
        'Venus': '#1f77b4', 'Earth': '#2ca02c',
        'Mars': '#d62728', 'Jupiter': '#ff7f0e',
    }

    plotted_labels = []  # (a, beta) of already-labelled curves to avoid clutter

    for planet_name in label_planets:
        a_p = PLANET_A[planet_name]
        color = color_map[planet_name]
        for p in range(1, 8):
            for q in range(1, 8):
                if p == q:
                    continue
                order = abs(p - q)
                if order > 2:
                    continue

                a_curve = a_p * (q / p) ** (2.0 / 3.0) * (1.0 - beta_fine) ** (1.0 / 3.0)

                # Only plot if curve passes through visible area
                in_range = (a_curve >= config['a_range'][0]) & (a_curve <= config['a_range'][1])
                if not np.any(in_range):
                    continue

                lw = 1.0 if order == 1 else 0.6
                alpha = 0.7 if order == 1 else 0.4
                ls = '-' if order == 1 else '--'

                ax.plot(a_curve, beta_fine, color=color, linewidth=lw,
                        alpha=alpha, linestyle=ls)

                # Label at beta=0 (bottom of curve)
                a_at_zero = a_p * (q / p) ** (2.0 / 3.0)
                if config['a_range'][0] <= a_at_zero <= config['a_range'][1]:
                    too_close = any(abs(a_at_zero - al) < 0.04 for al, _ in plotted_labels)
                    if not too_close and order <= 1:
                        label = f"{p}:{q} {planet_name[0]}"
                        ax.text(a_at_zero, -0.03, label,
                                ha='center', va='top', fontsize=5,
                                color=color, rotation=90, clip_on=False)
                        plotted_labels.append((a_at_zero, 0))

    # --- Planet locations (vertical lines at beta=0, curves inward at higher beta) ---
    for planet_name, planet_a in PLANET_A.items():
        if config['a_range'][0] <= planet_a <= config['a_range'][1]:
            ax.axvline(planet_a, color='black', linewidth=1.2, alpha=0.3, linestyle=':')
            ax.text(planet_a, config['beta_range'][1] + 0.02, planet_name[:3],
                    ha='center', va='bottom', fontsize=8, fontweight='bold', clip_on=False)

    ax.set_xlabel('Semimajor Axis (AU)', fontsize=13)
    ax.set_ylabel('Sail Lightness beta', fontsize=13)
    ax.set_title(
        f'Orbital Stability vs Sail Lightness (MEGNO, {config["integration_time"]:.0f} yr, '
        f'i={config["inc_deg"]:.0f} deg, e={config["e_test"]})',
        fontsize=14, fontweight='bold',
    )
    ax.set_xlim(config['a_range'])
    ax.set_ylim(config['beta_range'])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=1.0, label='Venus MMR'),
        Line2D([0], [0], color='#2ca02c', lw=1.0, label='Earth MMR'),
        Line2D([0], [0], color='#d62728', lw=1.0, label='Mars MMR'),
        Line2D([0], [0], color='#ff7f0e', lw=1.0, label='Jupiter MMR'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if save:
        prefix = config['output_prefix']
        for ext in ['png', 'pdf']:
            fname = f'{prefix}_megno_map.{ext}'
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"  Saved: {fname}")

    return fig


def main_ab():
    """Main entry point for (a, beta) mode."""
    config = CONFIG_AB
    print("=" * 70)
    print("  MEGNO Stability Map: (a, beta) mode")
    print("=" * 70)
    print()
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    n_workers = config['n_workers'] if config['n_workers'] > 0 else cpu_count()
    print(f"  detected cores: {cpu_count()}, will use: {n_workers}")
    print()

    # ---- Diagnostic ----
    beta_diag = min(0.3, config['beta_range'][1])
    print(f"Running diagnostic (a=1.5 AU, i={config['inc_deg']}deg, beta={beta_diag})...")
    try:
        global _LIGHTNESS
        _LIGHTNESS = beta_diag
        sim = create_solar_system_sim()
        sim.integrator = "whfast"
        sim.dt = 0.05
        sim.add(m=0.0, a=1.5, e=0.0, inc=np.radians(config['inc_deg']),
                Omega=0, omega=0, M=0)
        tp = sim.particles[sim.N - 1]
        scale = (1.0 - beta_diag) ** 0.5
        tp.vx *= scale; tp.vy *= scale; tp.vz *= scale
        sim.move_to_com()
        sim.additional_forces = _radiation_force
        sim.force_is_velocity_dependent = 0
        sim.init_megno()
        sim.integrate(100.0)
        print(f"  MEGNO after 100 yr:  {sim.megno():.4f}")
        sim.integrate(1000.0)
        print(f"  MEGNO after 1000 yr: {sim.megno():.4f}")
        print("  Diagnostic PASSED")
    except Exception as ex:
        print(f"  Diagnostic FAILED: {type(ex).__name__}: {ex}")
        return
    print()

    # ---- Check for existing results ----
    npz_final = f"{config['output_prefix']}_data.npz"
    ckpt_path = f"{config['output_prefix']}_checkpoint.npz"
    ckpt_meta = f"{config['output_prefix']}_checkpoint_meta.json"

    if os.path.exists(npz_final):
        print(f"Found completed results: {npz_final}")
        print("  (p) Re-plot only")
        print("  (r) Re-compute from scratch")
        print("  (c) Continue from checkpoint (if available)")
        resp = input("Choice [p/r/c]: ").strip().lower()
        if resp == 'p':
            data = np.load(npz_final, allow_pickle=True)
            plot_ab_map(data['a_values'], data['beta_values'], data['megno_map'],
                        config, save=True)
            plt.show()
            return
        elif resp == 'r':
            for f in [ckpt_path, ckpt_meta]:
                if os.path.exists(f):
                    os.remove(f)
    elif os.path.exists(ckpt_path):
        print(f"Found checkpoint: {ckpt_path}")
        print("  (c) Continue from checkpoint")
        print("  (r) Re-compute from scratch")
        resp = input("Choice [c/r]: ").strip().lower()
        if resp == 'r':
            for f in [ckpt_path, ckpt_meta]:
                if os.path.exists(f):
                    os.remove(f)

    # ---- Compute ----
    a_values, beta_values, megno_map = compute_ab_map(config)

    # Save final
    print("\nSaving final data...")
    np.savez(npz_final, a_values=a_values, beta_values=beta_values,
             megno_map=megno_map, config=config)
    print(f"  Saved: {npz_final}")

    # Plot
    print("\nCreating (a, beta) stability map...")
    plot_ab_map(a_values, beta_values, megno_map, config)

    # Statistics
    valid = megno_map[np.isfinite(megno_map)]
    if len(valid) > 0:
        print(f"\nStatistics ({len(valid)} valid points of {megno_map.size}):")
        print(f"  Mean MEGNO:  {np.mean(valid):.3f}")
        print(f"  Median MEGNO: {np.median(valid):.3f}")
        print(f"  Chaotic (>2): {np.sum(valid > 2) / len(valid) * 100:.1f}%")
        print(f"  Regular (<=2): {np.sum(valid <= 2) / len(valid) * 100:.1f}%")

    # Clean up checkpoints
    for f in [ckpt_path, ckpt_meta]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Cleaned up: {f}")

    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'ab':
        main_ab()
    else:
        print("Select mode:")
        print("  (1) Standard (a, i) stability map")
        print("  (2) Sail lightness (a, beta) map")
        resp = input("Choice [1/2]: ").strip()
        if resp == '2':
            main_ab()
        else:
            main()
