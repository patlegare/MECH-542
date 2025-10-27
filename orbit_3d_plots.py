
"""
- Propagate 2 orbital periods from each TLE's epoch for a smooth ellipse.
- Figures have equal aspect and Earth sphere rendered for context.
"""

import os
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from sgp4.api import Satrec, jday
except Exception as e:
    raise SystemExit(
        "This script requires the 'sgp4' package.\n"
        "Install it with: pip install sgp4\n"
        f"Import error was: {e}"
    )


# ----------------------------
# Helpers
# ----------------------------
def read_tle_file(path: str) -> List[Tuple[str, str]]:
    """Read a text file containing repeated TLE line1/line2 pairs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"TLE file not found: {path}")
    lines = [ln.strip() for ln in open(path, "r", encoding="utf-8") if ln.strip()]
    pairs = []
    i = 0
    while i < len(lines) - 1:
        l1 = lines[i]
        l2 = lines[i + 1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            pairs.append((l1, l2))
            i += 2
        else:
            # If the data contains duplicates or repeats, try to realign by one.
            i += 1
    if not pairs:
        raise ValueError(f"No TLE pairs found in {path}")
    return pairs


def tle_to_satrec(tle_pair: Tuple[str, str]) -> Satrec:
    l1, l2 = tle_pair
    return Satrec.twoline2rv(l1, l2)


def orbit_period_minutes(sat: Satrec) -> float:
    """Return orbital period (minutes) using mean motion from TLE (no_kozai in rad/min)."""
    # no_kozai is radians / minute; period T = 2*pi / n
    n_rad_per_min = sat.no_kozai
    return float(2.0 * np.pi / n_rad_per_min)


def propagate_arc(sat: Satrec, minutes: float, step_min: float = 0.5) -> np.ndarray:
    """
    Propagate in ECI over [0, minutes] from the TLE epoch with 'step_min' resolution.
    Returns Nx3 array in kilometers (TEME frame from SGP4; treat as ECI for visualization).
    """
    # Epoch from Satrec (UTC)
    jd, fr = sat.jdsatepoch, sat.jdsatepochF
    # Build times in minutes since epoch
    tsince = np.arange(0.0, minutes + step_min, step_min, dtype=float)
    xs, ys, zs = [], [], []
    for dt_min in tsince:
        e, r, v = sat.sgp4(jd, fr + dt_min / (24.0 * 60.0))
        if e != 0:
            # Skip bad steps (rare with sensible spans)
            continue
        xs.append(r[0])
        ys.append(r[1])
        zs.append(r[2])
    return np.column_stack([xs, ys, zs])


def set_equal_aspect_3d(ax: plt.Axes, X: np.ndarray) -> None:
    """Equal aspect for 3D axes based on data bounds."""
    max_range = np.array([X[:, 0].ptp(), X[:, 1].ptp(), X[:, 2].ptp()]).max()
    mid_x = (X[:, 0].max() + X[:, 0].min()) / 2.0
    mid_y = (X[:, 1].max() + X[:, 1].min()) / 2.0
    mid_z = (X[:, 2].max() + X[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)


def draw_earth(ax: plt.Axes, radius_km: float = 6378.137, n: int = 60) -> None:
    """Render a wireframe Earth for context (WGS84 equatorial radius)."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.3, alpha=0.5)


def render_orbit_png(
    tle_path: str,
    out_png: str,
    title: str,
    arc_periods: float = 2.0,
    step_min: float = 0.5,
    dpi: int = 300,
    transparent: bool = True,
) -> Tuple[float, float]:
    """
    Use the latest TLE in file, propagate ~arc_periods of the orbit, and save a 3D figure.
    Returns (inclination_deg, period_min) for optional captioning.
    """
    pairs = read_tle_file(tle_path)
    l1, l2 = pairs[-1]  # latest
    sat = tle_to_satrec((l1, l2))

    T_min = orbit_period_minutes(sat)
    minutes = arc_periods * T_min

    # Propagate positions
    r_eci = propagate_arc(sat, minutes, step_min=step_min)

    # Extract inclination from TLE line 2 (columns fixed-width in TLE standard)
    # Line 2 format: col 9-16 Inclination (degrees)
    try:
        inc_deg = float(l2[8:16])
    except Exception:
        inc_deg = np.nan

    # --- Plot ---
    fig = plt.figure(figsize=(11, 8))  # Large canvas; export is DPI-controlled
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r_eci[:, 0], r_eci[:, 1], r_eci[:, 2])
    draw_earth(ax)
    set_equal_aspect_3d(ax, r_eci)

    ax.set_title(title)
    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_zlabel("ECI Z (km)")

    # Light grid; neutral look
    ax.grid(True)

    # Save
    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, transparent=transparent, bbox_inches="tight")
    plt.close(fig)

    return inc_deg, T_min


def render_combined_png(
    tle_path_a: str,
    tle_path_b: str,
    out_png: str,
    title: str = "3D Orbits (Latest TLE)",
    arc_periods: float = 2.0,
    step_min: float = 0.5,
    dpi: int = 300,
    transparent: bool = True,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Render both orbits in a single 3D figure; returns ((incA, TA), (incB, TB))."""
    pairs_a = read_tle_file(tle_path_a)
    pairs_b = read_tle_file(tle_path_b)

    l1a, l2a = pairs_a[-1]
    l1b, l2b = pairs_b[-1]

    sat_a = tle_to_satrec((l1a, l2a))
    sat_b = tle_to_satrec((l1b, l2b))

    Ta = orbit_period_minutes(sat_a)
    Tb = orbit_period_minutes(sat_b)

    r_a = propagate_arc(sat_a, arc_periods * Ta, step_min=step_min)
    r_b = propagate_arc(sat_b, arc_periods * Tb, step_min=step_min)

    try:
        inc_a = float(l2a[8:16])
    except Exception:
        inc_a = np.nan
    try:
        inc_b = float(l2b[8:16])
    except Exception:
        inc_b = np.nan

    # Plot
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(r_a[:, 0], r_a[:, 1], r_a[:, 2], label="NORAD 37869")
    ax.plot(r_b[:, 0], r_b[:, 1], r_b[:, 2], label="NORAD 63130")

    draw_earth(ax)
    # Make aspect equal using both orbits together
    X = np.vstack([r_a, r_b])
    set_equal_aspect_3d(ax, X)

    ax.set_title(title)
    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_zlabel("ECI Z (km)")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, transparent=transparent, bbox_inches="tight")
    plt.close(fig)

    return (inc_a, Ta), (inc_b, Tb)


def main():
    tle_37869 = "sat000037869.txt"
    tle_63130 = "sat000063130.txt"

    # If running in the same directory as the files, fine; otherwise adjust paths.
    if not os.path.exists(tle_37869) and os.path.exists(os.path.join("..", tle_37869)):
        tle_37869 = os.path.join("..", tle_37869)
    if not os.path.exists(tle_63130) and os.path.exists(os.path.join("..", tle_63130)):
        tle_63130 = os.path.join("..", tle_63130)

    # Individual plots
    inc1, T1 = render_orbit_png(
        tle_37869, "orbit_37869.png", "NORAD 37869 – 3D Orbit (Latest TLE)"
    )
    inc2, T2 = render_orbit_png(
        tle_63130, "orbit_63130.png", "NORAD 63130 – 3D Orbit (Latest TLE)"
    )

    # Combined plot
    (inc_a, Ta), (inc_b, Tb) = render_combined_png(
        tle_37869, tle_63130, "orbits_combined.png"
    )

    # Console summary (useful for captioning in Canva)
    print("---- Summary ----")
    print(f"NORAD 37869: inclination ~ {inc1:.3f} deg | period ~ {T1:.2f} min")
    print(f"NORAD 63130: inclination ~ {inc2:.3f} deg | period ~ {T2:.2f} min")
    print("Saved: orbit_37869.png, orbit_63130.png, orbits_combined.png")


if __name__ == "__main__":
    main()
