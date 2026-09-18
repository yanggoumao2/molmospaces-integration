from __future__ import annotations

import h5py
import numpy as np
from types import SimpleNamespace

from src.pnp_bimanual_yam.env_interface import (
    apply_linear_xy_offset_blend,
    nearest_monotonic_source_index,
)
from src.pnp_bimanual_yam.task_spec import (
    LEFT_ARM_WINDOW,
    LEFT_SUBTASKS,
    RIGHT_ARM_WINDOW,
    RIGHT_SUBTASKS,
    build_task_spec,
)
from src.pnp_bimanual_yam.run_datagen import _apply_attempt_transfer_blend


RECORDER = "runtime/mimicgen_pick_and_place/bimanual_v1/contract_xy005_gate_20260914/record_corrected_xy005.py"


WORK = "runtime/mimicgen_pick_and_place/bimanual_v1/contract_xy005_gate_20260914"


def _horizontal_travel(arm: str, start_abs: int, end_abs: int) -> float:
    window_start = RIGHT_ARM_WINDOW[0] if arm == "right" else LEFT_ARM_WINDOW[0]
    with h5py.File(f"{WORK}/source_{arm}.hdf5", "r") as source:
        xyz = np.asarray(source["data/demo_0/datagen_info/target_pose"][:, :3, 3])
    start = start_abs - window_start
    end = end_abs - window_start
    return float(np.linalg.norm(xyz[end - 1, :2] - xyz[start, :2]))


def test_transfer_offset_is_distributed_across_horizontal_motion() -> None:
    assert RIGHT_SUBTASKS[2][3] == 3652
    assert LEFT_SUBTASKS[2][3] == 9212
    poses = np.repeat(np.eye(4)[None], 10, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.3, 10)
    blended = apply_linear_xy_offset_blend(poses, 2, 9, np.array([0.07, -0.04]))
    correction = blended[:, :2, 3] - poses[:, :2, 3]
    assert np.allclose(correction[2], 0.0)
    assert np.allclose(correction[8], [0.07, -0.04])
    assert np.max(np.linalg.norm(np.diff(correction[2:9], axis=0), axis=1)) < 0.02


def test_source_match_cannot_jump_to_later_retreat_loop() -> None:
    outbound = np.arange(11, dtype=float)
    retreat = np.arange(9, -1, -1, dtype=float)
    source = np.column_stack([np.concatenate([outbound, retreat]), np.zeros(21), np.zeros(21)])
    assert nearest_monotonic_source_index(source, np.array([5.1, 0.0, 0.0]), 4, 4) == 5
    assert nearest_monotonic_source_index(source, np.array([4.9, 0.0, 0.0]), 6, 4) >= 6


def test_transfer_blend_stops_before_plate_reference_boundary() -> None:
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, 0, 3] = np.arange(6, dtype=float)
    blended = apply_linear_xy_offset_blend(poses, 1, 4, np.array([0.3, 0.2]))
    correction = blended[:, :2, 3] - poses[:, :2, 3]
    assert np.allclose(correction[3], [0.3, 0.2])
    assert np.allclose(correction[4], [0.0, 0.0])


def test_recorder_blends_the_dataset_attribute_used_by_generate() -> None:
    source = open(RECORDER, encoding="utf-8").read()
    assert "right_gen.src_dataset_infos[0].target_pose" in source
    assert "left_gen.src_dataset_infos[0].target_pose" in source
    assert "right_gen.datagen_infos[0].target_pose" not in source
    assert "left_gen.datagen_infos[0].target_pose" not in source


def test_recorder_uses_exclusive_carry_start_indices() -> None:
    source = open(RECORDER, encoding="utf-8").read()
    assert "target_pose, 1915, 2838, right_delta" in source
    assert "target_pose, 2513, 3436, left_delta" in source


def test_collector_rebuilds_blend_from_base_for_each_layout() -> None:
    base_right = np.repeat(np.eye(4)[None], 2838, axis=0)
    base_left = np.repeat(np.eye(4)[None], 3436, axis=0)
    base_right_xyz = np.zeros((2838, 3), dtype=float)
    base_left_xyz = np.zeros((3436, 3), dtype=float)
    right_gen = SimpleNamespace(src_dataset_infos=[SimpleNamespace(target_pose=base_right)])
    left_gen = SimpleNamespace(src_dataset_infos=[SimpleNamespace(target_pose=base_left)])
    right_iface = SimpleNamespace(_src_eef_pos=None)
    left_iface = SimpleNamespace(_src_eef_pos=None)

    _apply_attempt_transfer_blend(
        right_gen,
        left_gen,
        right_iface,
        left_iface,
        base_right,
        base_left,
        base_right_xyz,
        base_left_xyz,
        {"plate": [0.08, 0.0], "potato": [0.0, 0.0], "tomato": [0.0, 0.0]},
    )
    _apply_attempt_transfer_blend(
        right_gen,
        left_gen,
        right_iface,
        left_iface,
        base_right,
        base_left,
        base_right_xyz,
        base_left_xyz,
        {"plate": [-0.06, 0.0], "potato": [0.0, 0.0], "tomato": [0.0, 0.0]},
    )

    assert np.isclose(right_gen.src_dataset_infos[0].target_pose[2837, 0, 3], -0.06)
    assert np.isclose(right_iface._src_eef_pos[2837, 0], -0.06)
    assert np.isclose(base_right[2837, 0, 3], 0.0)
