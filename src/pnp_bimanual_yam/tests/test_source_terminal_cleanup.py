from __future__ import annotations

import ast
import os
from pathlib import Path

import h5py
import numpy as np

from src.pnp_bimanual_yam.env_wrapper import food_success_components


DEFAULT_SOURCE = Path(
    "work/current/floorplan1_yam_potato_grasp_place_20260831_2038/"
    "codex_tasks/yam_food_bimanual_20260908_1754/"
    "execution_attempt_20260914_023902/source_demo_enriched.h5"
)
LEFT_WINDOW_START = 5777
OPEN_CTRL = -0.0475
TASK_SPEC_PATH = Path(__file__).resolve().parents[1] / "task_spec.py"
RUN_DATAGEN_PATH = Path(__file__).resolve().parents[1] / "run_datagen.py"
CORRECTED_ATTEMPT = "execution_attempt_20260914_023902"


def _source_path() -> Path:
    return Path(os.environ.get("YAM_SOURCE_HDF5", DEFAULT_SOURCE))


def _literal_assignment(name: str):
    tree = ast.parse(TASK_SPEC_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment: {name}")


def _path_assignment(path: Path, name: str) -> Path:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        assert isinstance(node.value, ast.Call) and node.value.args
        return Path(ast.literal_eval(node.value.args[0]))
    raise AssertionError(f"missing Path assignment: {name}")


def test_left_source_returns_near_its_pre_task_pose_after_release() -> None:
    with h5py.File(_source_path(), "r") as source:
        group = source["traj_0/mimicgen_yam"]
        left_eef = np.asarray(group["ee_pose/left/pos"], dtype=np.float64)
        left_command = np.asarray(group["replay_commands/left_arm"], dtype=np.float64)

    eef_home_error = np.linalg.norm(left_eef[-1] - left_eef[LEFT_WINDOW_START])
    command_home_error = np.max(np.abs(left_command[-1] - left_command[LEFT_WINDOW_START]))

    assert eef_home_error < 0.03, f"left EEF remains {eef_home_error:.4f} m from home"
    assert command_home_error < 0.10, (
        f"left command remains {command_home_error:.4f} rad from its pre-task pose"
    )


def test_left_source_finishes_with_gripper_open() -> None:
    with h5py.File(_source_path(), "r") as source:
        left_gripper = np.asarray(
            source["traj_0/mimicgen_yam/replay_commands/left_gripper"],
            dtype=np.float64,
        )

    assert np.allclose(left_gripper[-1], OPEN_CTRL, atol=1e-6)


def test_left_task_spec_window_includes_terminal_cleanup() -> None:
    with h5py.File(_source_path(), "r") as source:
        source_steps = source["traj_0/mimicgen_yam/replay_commands/left_arm"].shape[0]

    left_arm_window = _literal_assignment("LEFT_ARM_WINDOW")
    left_subtasks = _literal_assignment("LEFT_SUBTASKS")
    expected_end = source_steps - 1
    assert left_arm_window == (LEFT_WINDOW_START, expected_end)
    assert left_subtasks[-1][3] == expected_end


def test_datagen_default_uses_corrected_source() -> None:
    run_datagen_source = RUN_DATAGEN_PATH.read_text(encoding="utf-8")
    assert f"{CORRECTED_ATTEMPT}/source_demo_enriched.h5" in run_datagen_source

    # The default is repository-relative and may be overridden through
    # YAM_SOURCE_HDF5; avoid requiring the rental machine's absolute path.
    assert "YAM_SOURCE_HDF5" in run_datagen_source
    assert "source_demo_enriched.h5" in run_datagen_source


def test_success_contract_requires_release_support_and_no_drop() -> None:
    contacts = [
        {
            "body1_id": 11,
            "body2_id": 20,
            "geom1_id": 101,
            "geom2_id": 201,
            "body1_name": "food_potato",
            "body2_name": "plate",
        },
    ]
    result = food_success_components(
        contacts,
        food_body_id=10,
        fingertip_geom_ids={301, 302},
        receptacle_token="plate",
        not_dropped=True,
        food_root_body_id=10,
        food_body_ids={10, 11},
    )
    assert result == {
        "released": True,
        "plate_supported": True,
        "not_dropped": True,
        "success": True,
    }

    contacts.append(
        {
            "body1_id": 11,
            "body2_id": 30,
            "geom1_id": 301,
            "geom2_id": 401,
            "body1_name": "food_potato",
            "body2_name": "robot_0/right_left_tip_collider_0",
        }
    )
    result = food_success_components(
        contacts,
        food_body_id=10,
        fingertip_geom_ids={301, 302},
        receptacle_token="plate",
        not_dropped=True,
        food_root_body_id=10,
        food_body_ids={10, 11},
    )
    assert result["released"] is False
    assert result["success"] is False


def test_success_contract_uses_post_settle_not_velocity_threshold() -> None:
    contacts = [
        {
            "body1_id": 11,
            "body2_id": 20,
            "geom1_id": 101,
            "geom2_id": 201,
            "body1_name": "food_potato",
            "body2_name": "plate",
        },
    ]
    result = food_success_components(
        contacts,
        food_body_id=10,
        fingertip_geom_ids={301, 302},
        receptacle_token="plate",
        not_dropped=True,
        food_root_body_id=10,
        food_body_ids={10, 11},
    )
    assert result["released"] is True
    assert result["plate_supported"] is True
    assert result["not_dropped"] is True
    assert result["success"] is True
