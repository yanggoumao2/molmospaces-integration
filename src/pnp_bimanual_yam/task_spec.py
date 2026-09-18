"""MG_TaskSpec for FloorPlan1 bimanual YAM food-placement.

Source demo has 10 subtasks over 11312 steps. Sequential (DexMimicGen) mode
splits them into two per-arm episodes:
- Right (subtasks 0..4): approach/grasp/lift potato, carry to plate, release_and_stow
- Left  (subtasks 5..9): approach/grasp/lift tomato, carry to plate, release_and_stow

Subtask boundaries below are aligned to the corrected source demo
execution_attempt_20260914_023902. When converting the source demo into a
per-arm MimicGen HDF5, indices are re-based to the arms active window
(right: 815..5777; left: 5777..11311).
"""

from __future__ import annotations

from src.pnp_bimanual_yam import _shims as _  # install robosuite/gdown shims
from mimicgen.configs.task_spec import MG_TaskSpec

# (object_ref, term_signal_name, source_start_abs, source_end_abs)
RIGHT_SUBTASKS = [
    ("potato", "right_pregrasp_done", 815, 1772),
    ("potato", "right_grasp_done", 1772, 2329),
    ("potato", "right_lift_done", 2329, 3652),
    ("plate", "right_carry_done", 3652, 3852),
    ("plate", None, 3852, 5777),
]
LEFT_SUBTASKS = [
    ("tomato", "left_pregrasp_done", 5777, 7332),
    ("tomato", "left_grasp_done", 7332, 7819),
    ("tomato", "left_lift_done", 7819, 9212),
    ("plate", "left_carry_done", 9212, 9412),
    ("plate", None, 9412, 11311),
]
RIGHT_ARM_WINDOW = (815, 5777)  # source-demo absolute [start, end)
LEFT_ARM_WINDOW = (5777, 11311)


def build_task_spec(
    arm: str, *, noise: float = 0.0, num_interpolation_steps: int = 40
) -> MG_TaskSpec:
    """Build a 5-subtask MG_TaskSpec for one arm.

    selection_strategy is forced to random because we generate from a single
    source demo (no demo pool). subtask_term_offset_range=(0,0) keeps subtask
    boundaries exact; increase later once the smoke gate is stable.
    """
    if arm == "right":
        subtasks = RIGHT_SUBTASKS
    elif arm == "left":
        subtasks = LEFT_SUBTASKS
    else:
        raise ValueError(f"arm must be right|left, got {arm!r}")
    spec = MG_TaskSpec()
    for object_ref, signal, _s, _e in subtasks:
        spec.add_subtask(
            object_ref=object_ref,
            subtask_term_signal=signal,
            subtask_term_offset_range=(0, 0),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=float(noise),
            num_interpolation_steps=int(num_interpolation_steps),
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
    return spec


def arm_window(arm: str) -> tuple[int, int]:
    return RIGHT_ARM_WINDOW if arm == "right" else LEFT_ARM_WINDOW


def relative_subtask_boundaries(arm: str) -> list[tuple[str, str | None, int, int]]:
    """Return subtask (object_ref, signal, rel_start, rel_end) with indices
    re-based to the arms active window."""
    window_start, _ = arm_window(arm)
    subtasks = RIGHT_SUBTASKS if arm == "right" else LEFT_SUBTASKS
    return [(o, s, abs_s - window_start, abs_e - window_start) for o, s, abs_s, abs_e in subtasks]
