"""Convert the single FloorPlan1 bimanual YAM source demo into two MimicGen-format HDF5s.

Input:
    source_demo_enriched.h5 with the layout produced by
    execution_attempt_20260911_121406 (6-DoF EE, per-arm arm_qpos, per-arm
    gripper, per-object pose, replay_commands).

Output (one file per arm):
    data/
      demo_0/
        actions              (T_arm, 7)   [6 arm joints + 1 gripper]
        datagen_info/
          eef_pose           (T_arm, 4, 4) world-frame
          object_poses/
            <object>         (T_arm, 4, 4)
          subtask_term_signals/
            <signal>         (T_arm,)  uint8
          target_pose        (T_arm, 4, 4)  eef_pose[t+1] (last step repeats)
          gripper_action     (T_arm, 1)

MimicGen only uses subtask_term_signals for source-side subtask boundary
detection (first 0->1 transition). We construct step-function indicators so
that each signal transitions to 1 exactly at the recorded subtask end index
(re-based to the arm window).

Actions and target_pose are recorded for completeness. The bimanual sequential
run reuses these but only relies on eef_pose + object_poses + subtask signals
for the T_new * T_src^-1 transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from src.pnp_bimanual_yam._scene import build_scene

from src.pnp_bimanual_yam.task_spec import (
    LEFT_SUBTASKS,
    RIGHT_SUBTASKS,
    arm_window,
)


OBJECTS = ("plate", "potato", "tomato")


def _quat_wxyz_to_rot_mat(quat: np.ndarray) -> np.ndarray:
    """Convert (T,4) wxyz quaternion into (T,3,3) rotation matrices."""
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-12, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _assemble_homogeneous(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """(T,3) + (T,3,3) -> (T,4,4)."""
    T = pos.shape[0]
    H = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    H[:, :3, :3] = rot
    H[:, :3, 3] = pos
    return H


def _ee_hmat(grp: h5py.Group, arm: str) -> np.ndarray:
    """Build EEF poses from the arm commands that passed identity replay.

    The enriched recorder's ``ee_pose`` channel is not frame-aligned with
    ``replay_commands`` at several grasp-critical points. MimicGen combines
    EEF poses with gripper commands, so both streams must originate from the
    same executed control contract.
    """
    arm_commands = np.asarray(grp[f"replay_commands/{arm}_arm"], dtype=np.float64)
    if arm_commands.ndim != 2 or arm_commands.shape[1] != 6:
        raise ValueError(
            f"replay_commands/{arm}_arm must have shape (T, 6), got {arm_commands.shape}"
        )

    model, data, arms, _foods = build_scene()
    adapter = arms[arm]
    eef = np.tile(np.eye(4, dtype=np.float64), (len(arm_commands), 1, 1))
    for index, command in enumerate(arm_commands):
        data.qpos[adapter.qaddrs] = command
        mujoco.mj_forward(model, data)
        eef[index, :3, :3] = data.site_xmat[adapter.grasp_site].reshape(3, 3)
        eef[index, :3, 3] = data.site_xpos[adapter.grasp_site]
    return eef


def _obj_hmat(grp: h5py.Group, obj: str) -> np.ndarray:
    # MimicGen applies T_new * T_src^-1 * T_src_ee. Any rotation delta between
    # source and runtime object frames would rotate the target EE around the
    # object center. Since our IK is position-only and the source demos object
    # orientation is not meaningful for pick-and-place placement gate, we force
    # object rotation to identity here (and mirror in env_wrapper.get_object_poses_world).
    pos = np.asarray(grp[f"object_pose/{obj}/pos"], dtype=np.float64)
    T = pos.shape[0]
    rot = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    return _assemble_homogeneous(pos, rot)


def _term_signals(arm: str, T_arm: int) -> dict[str, np.ndarray]:
    """Build step-function indicators aligned with the arms subtask boundaries.

    RIGHT_SUBTASKS/LEFT_SUBTASKS use absolute source-demo indices. Re-base each
    subtask end index to the arm window, and set indicator[end:] = 1. MimicGen
    parse_source_dataset only uses the first 0->1 transition, so a monotone
    step function is sufficient and unambiguous.
    """
    subtasks = RIGHT_SUBTASKS if arm == "right" else LEFT_SUBTASKS
    win_start, _ = arm_window(arm)
    signals: dict[str, np.ndarray] = {}
    for _obj, signal, _abs_start, abs_end in subtasks:
        if signal is None:
            continue
        rel_end = abs_end - win_start
        assert 0 < rel_end <= T_arm, (
            f"arm {arm}: subtask end {abs_end} maps to rel {rel_end} outside [0, {T_arm}]"
        )
        sig = np.zeros(T_arm, dtype=np.uint8)
        sig[rel_end:] = 1
        signals[signal] = sig
    return signals


def _build_actions_and_gripper(
    grp: h5py.Group, arm: str, win: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (actions [T,7], gripper_action [T,1]) for the arm window."""
    s, e = win
    arm_cmd = np.asarray(grp[f"replay_commands/{arm}_arm"][s:e], dtype=np.float32)  # (T, 6)
    grip_cmd = np.asarray(grp[f"replay_commands/{arm}_gripper"][s:e], dtype=np.float32)  # (T, 2)
    # collapse two-finger gripper command to a scalar (mean); MimicGen only needs
    # a per-timestep gripper actuation, and both fingers command in lockstep here.
    grip_scalar = grip_cmd.mean(axis=1, keepdims=True).astype(np.float32)  # (T, 1)
    actions = np.concatenate([arm_cmd, grip_scalar], axis=1).astype(np.float32)  # (T, 7)
    return actions, grip_scalar


def _target_pose_from_next_eef(eef_hmat: np.ndarray) -> np.ndarray:
    """target_pose[t] = eef_pose[t+1]; last step repeats."""
    tp = np.empty_like(eef_hmat)
    tp[:-1] = eef_hmat[1:]
    tp[-1] = eef_hmat[-1]
    return tp


def convert_arm(
    src_path: Path,
    out_path: Path,
    arm: str,
    *,
    traj_key: str = "traj_0",
    demo_key: str = "demo_0",
) -> dict:
    with h5py.File(src_path, "r") as f:
        grp = f[traj_key]["mimicgen_yam"]
        win = arm_window(arm)
        s, e = win
        T_arm = e - s
        # per-arm EEF (world-frame, from source demo)
        eef_all = _ee_hmat(grp, arm)
        eef = eef_all[s:e]
        # object poses (all objects present, regardless of which arm is active).
        object_poses = {obj: _obj_hmat(grp, obj)[s:e] for obj in OBJECTS}
        # arm-window subtask termination signals
        term_signals = _term_signals(arm, T_arm)
        # actions
        actions, gripper_action = _build_actions_and_gripper(grp, arm, win)
        # target pose = next eef; last repeats
        target_pose = _target_pose_from_next_eef(eef)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as g:
        data_grp = g.create_group("data")
        data_grp.attrs["arm"] = arm
        data_grp.attrs["source_hdf5"] = str(src_path)
        data_grp.attrs["window_start_abs"] = int(s)
        data_grp.attrs["window_end_abs"] = int(e)
        data_grp.attrs["T_arm"] = int(T_arm)
        demo_grp = data_grp.create_group(demo_key)
        demo_grp.create_dataset("actions", data=actions)
        dg = demo_grp.create_group("datagen_info")
        dg.create_dataset("eef_pose", data=eef.astype(np.float32))
        dg.create_dataset("target_pose", data=target_pose.astype(np.float32))
        dg.create_dataset("gripper_action", data=gripper_action.astype(np.float32))
        obj_grp = dg.create_group("object_poses")
        for name, mat in object_poses.items():
            obj_grp.create_dataset(name, data=mat.astype(np.float32))
        sig_grp = dg.create_group("subtask_term_signals")
        for name, sig in term_signals.items():
            sig_grp.create_dataset(name, data=sig)
    return {
        "arm": arm,
        "T_arm": int(T_arm),
        "window": [int(s), int(e)],
        "signals": sorted(term_signals),
        "objects": sorted(object_poses),
        "out_path": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="path to source_demo_enriched.h5")
    ap.add_argument("--out-right", required=True, type=Path)
    ap.add_argument("--out-left", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    right = convert_arm(args.src, args.out_right, arm="right")
    left = convert_arm(args.src, args.out_left, arm="left")
    manifest = {"right": right, "left": left}
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
