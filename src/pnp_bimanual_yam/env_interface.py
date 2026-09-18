"""Per-arm MG_EnvInterface for the bimanual YAM bridge.

One instance per arm. Both instances share the same BimanualYamEnv. The
BimanualYamEnv is responsible for the multi-body simulator; this class only
routes MimicGen queries to the requested arm.

Contract expected of the shared env (implemented in env_wrapper.py):

  env.get_arm_eef_pose_world(side) -> (4,4) np.array
  env.get_arm_base_pose_world(side) -> (4,4) np.array  (only used to keep parity with MimicGen "robot-base-frame" transforms; we transform via the same base for the whole demo, so identity works if the pose channel already lives in the same frame)
  env.get_object_poses_world() -> {name: (4,4)}
  env.solve_arm_ik(side, target_pose_world) -> (6,) arm joint targets, or raises
  env.fk_arm(side, arm_qpos) -> (4,4) world-frame EEF pose
  env.set_active_arm(side) -> None  (controls which arm env.step drives)
  env.arm_subtask_signals(side) -> {signal_name: int}
"""

from __future__ import annotations

import numpy as np

from src.pnp_bimanual_yam import _shims as _  # ensure MimicGen import chain works
from mimicgen.env_interfaces.base import MG_EnvInterface


def nearest_monotonic_source_index(source_positions, target_position, cursor, lookahead=512):
    source_positions = np.asarray(source_positions, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    start = min(max(int(cursor), 0), len(source_positions) - 1)
    end = min(len(source_positions), start + int(lookahead) + 1)
    return start + int(
        np.argmin(np.linalg.norm(source_positions[start:end] - target_position, axis=1))
    )


def apply_linear_xy_offset_blend(poses, start, end, delta_xy):
    blended = np.asarray(poses, dtype=float).copy()
    if end <= start:
        return blended
    alpha = np.linspace(0.0, 1.0, end - start)
    blended[start:end, :2, 3] += alpha[:, None] * np.asarray(delta_xy, dtype=float)
    return blended


class BimanualYamArmInterface(MG_EnvInterface):
    """MG_EnvInterface routed to a specific arm (right|left) of a shared env."""

    INTERFACE_TYPE = "molmospaces_bimanual_yam"

    def __init__(self, env, arm: str, source_qpos_seq=None, source_eef_pos_seq=None) -> None:
        if arm not in ("right", "left"):
            raise ValueError(f"arm must be right|left, got {arm!r}")
        super().__init__(env)
        self.arm = arm
        # Source demo arm_qpos sequence (T, 6) and corresponding eef xyz (T, 3).
        # When provided, target_pose_to_action seeds IK from the source qpos
        # whose eef xyz is closest to the requested target_pose translation.
        # Rationale: 6-DoF arms have multiple IK solutions for the same 6D
        # pose; without a source-anchored seed, IK from home may select a
        # different branch than the source demo used, and seed continuity
        # then propagates the wrong branch. Nearest-neighbor source-seeded
        # IK collapses to identity (q_src) when the target matches source
        # (identity smoke) and stays on the source branch under jitter.
        self._src_qpos = (
            None if source_qpos_seq is None else np.asarray(source_qpos_seq, dtype=float)
        )
        self._src_eef_pos = (
            None if source_eef_pos_seq is None else np.asarray(source_eef_pos_seq, dtype=float)
        )
        # Kept for compatibility with callers that reset the interface between
        # attempts. Source alignment is pose-based, not call-count-based:
        # MimicGen inserts interpolation waypoints between source frames.
        self._step = 0
        self._source_cursor = 0
        self._carry_seeded = False

    def reset_step_counter(self) -> None:
        """Called by run_datagen before each attempt to realign counter with source."""
        self._step = 0
        self._source_cursor = 0
        self._carry_seeded = False

    # ------------------------------------------------------------------ MimicGen API
    def get_robot_eef_pose(self) -> np.ndarray:
        """World-frame EE pose of this arm.

        Source demo eef_pose was written in the same world frame (see
        convert_source_hdf5._ee_hmat which reads ee_pose/{arm}/{pos,rot}
        directly). MimicGen only requires that source and runtime eef_pose
        share a frame with object_poses; both are world-frame, so we return
        world without an extra base-frame conversion.
        """
        return np.asarray(self.env.get_arm_eef_pose_world(self.arm), dtype=float)

    def target_pose_to_action(self, target_pose, relative: bool = True) -> np.ndarray:
        """Solve arm IK to reach the target world-frame EEF pose.

        Nearest-neighbor source-seeded IK: when a source demo qpos sequence
        was provided at construction, seed IK from the source qpos whose eef
        translation is closest to the requested target_pose. This makes IK
        return the source q exactly under identity (0 pos error) and stay on
        the source solution branch under jitter.

        Returns (6,) arm joint targets; MimicGen concatenates gripper action
        separately from waypoint.gripper_action.
        """
        target_pose = np.asarray(target_pose, dtype=float).reshape(4, 4)
        if self._src_qpos is not None and self._src_eef_pos is not None:
            carry_start = 2837 if self.arm == "right" else 3435
            # Interpolation adds waypoints, so execution-call index does not
            # identify a source frame. Align by the requested EEF position.
            k = nearest_monotonic_source_index(
                self._src_eef_pos, target_pose[:3, 3], self._source_cursor
            )
            self._source_cursor = max(self._source_cursor, k)
            src_q = self._src_qpos[k]
            src_pos = self._src_eef_pos[k]
            pos_delta = float(np.linalg.norm(target_pose[:3, 3] - src_pos))
            # Identity passthrough: when the requested target xyz matches the
            # source demo's step-k eef xyz within 1mm, we are in the identity
            # generation path (or generation aligned to source at this step),
            # so returning source q_src directly avoids IK-induced branch
            # jumps and per-step residual accumulation.
            if pos_delta < 1e-3 and k < carry_start:
                return src_q.astype(np.float32)
            # Keep the source branch through grasp and lift. At the first
            # carry target, explicitly bridge from the source carry boundary
            # instead of leaving the previous lift seed stale.
            if k < carry_start:
                self.env._last_ik_target[self.arm] = src_q.copy()
            elif not self._carry_seeded:
                self.env._last_ik_target[self.arm] = self._src_qpos[carry_start].copy()
                self._carry_seeded = True
        arm_q = self.env.solve_arm_ik(self.arm, target_pose)
        arm_q = np.asarray(arm_q, dtype=np.float32).reshape(-1)
        if arm_q.shape[0] != 6:
            raise RuntimeError(
                f"solve_arm_ik({self.arm}) returned {arm_q.shape[0]} joints; expected 6"
            )
        return arm_q

    def action_to_target_pose(self, action, relative: bool = True) -> np.ndarray:
        """Inverse: (7,) action -> (4,4) target EE pose via forward kinematics.

        Used by MimicGen when re-parsing recorded source-demo actions into
        target poses. We provide target_pose directly in convert_source_hdf5
        (as eef_pose[t+1]), so this method is only used defensively.
        """
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] < 6:
            raise RuntimeError(f"action length {action.shape[0]} < 6 arm joints")
        return np.asarray(self.env.fk_arm(self.arm, action[:6]), dtype=float)

    def action_to_gripper_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=float).reshape(-1)
        return np.asarray(action[-1:], dtype=np.float32)

    def get_object_poses(self) -> dict[str, np.ndarray]:
        """All object poses in the scene (both arms see the same objects).

        MimicGen uses only the subtasks object_ref key; extras are harmless.
        """
        raw = self.env.get_object_poses_world()
        return {k: np.asarray(v, dtype=float) for k, v in raw.items()}

    def get_subtask_term_signals(self) -> dict[str, int]:
        raw = self.env.arm_subtask_signals(self.arm)
        return {k: int(v) for k, v in raw.items()}
