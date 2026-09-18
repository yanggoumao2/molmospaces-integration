"""Sampling-domain configs for FloorPlan1 bimanual YAM datagen.

Produces sample_domain dicts consumed by BimanualYamEnv.reset(). Kept as a
separate module so the calibrated numbers (table z, empirical +/-15 cm
rectangle, plate/potato/tomato centers) live in one place and can be tuned
without touching env_wrapper.
"""

from __future__ import annotations

from typing import Any

# Calibrated on 2026-09-11 against source_demo_enriched.h5.
TABLE_Z_M = 0.9323

# Empirical rectangle from initial-domain calibration. Not derived from the arms
# reach disc alone; kept as (dx, dy) half-widths in meters.
EMPIRICAL_XY_HALF = (0.15, 0.15)

# Recorded initial xy centers from the source demo t=0 object poses. Update
# by re-reading source_demo_enriched.h5 if the scene changes.
INITIAL_CENTERS: dict[str, tuple[float, float]] = {
    "plate": (0.63, -1.72),
    "potato": (0.90, -1.70),
    "tomato": (0.85, -2.01),
}


def default_domain(seed: int | None = None) -> dict[str, Any]:
    """Return the initial-domain sample_domain used for smoke runs.

    All three objects randomize within +/- EMPIRICAL_XY_HALF around their
    recorded centers; z is pinned to the calibrated table plane.
    """
    return {
        "seed": seed,
        "objects": {
            label: {
                "xy_center": INITIAL_CENTERS[label],
                "xy_half": EMPIRICAL_XY_HALF,
                "z_fixed": TABLE_Z_M,
            }
            for label in INITIAL_CENTERS
        },
    }


def narrow_domain(seed: int | None = None, half: float = 0.05) -> dict[str, Any]:
    """Fallback narrow-domain config when hit-rate on default domain is low."""
    return {
        "seed": seed,
        "objects": {
            label: {
                "xy_center": INITIAL_CENTERS[label],
                "xy_half": (half, half),
                "z_fixed": TABLE_Z_M,
            }
            for label in INITIAL_CENTERS
        },
    }
