"""BimanualYamEnv: MimicGen-compatible env wrapping the FloorPlan1 double-YAM scene.

Contract for MimicGen DataGenerator.generate():
    env.reset()                          # sample new object placements
    env.get_state()   -> {"states": ...} # initial-state serialization
    env.get_observation() -> dict        # per-step observation (opaque to us)
    env.step(action)  -> (obs, r, done, info)
    env.is_success()  -> {"task": bool, ...}

Additional contract exposed to BimanualYamArmInterface (env_interface.py):
    env.get_arm_eef_pose_world(side) -> (4,4)
    env.get_object_poses_world()     -> {name: (4,4)}
    env.solve_arm_ik(side, target)   -> (6,) joint targets
    env.fk_arm(side, arm_qpos)       -> (4,4)
    env.set_active_arm(side)         -> None
    env.arm_subtask_signals(side)    -> {name: 0|1}
    env.action_dim                   -> int (== 7)

Action layout (per step, one arm active at a time):
    action.shape == (7,) = [6 arm joint targets, 1 gripper scalar]
    The active arm receives the arm/gripper command; the inactive arm holds
    its last commanded pose. `set_active_arm(side)` is called by the datagen
    runner before running that arms subtask block.

Design notes:
- Success gate is NOT downgraded relative to the attempt runner: we still
  require final `stable_and_supported` for both foods on the plate. We only
  strip the MP4/frame/audit debug that made single attempts 10 min.
- reset() re-samples potato/tomato/plate XY within a calibrated rectangle.
  Sampling domain and z-plane come from work log 2026-09-11:
     table z = 0.9323 m; per-object XY jitter within +/- 15 cm.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from src.pnp_bimanual_yam._scene import (
    OPEN_CTRL,
    ArmAdapter,
    build_scene,
    contact_pairs,
    solve_ik,
)


TABLE_Z = 0.9323
DEFAULT_XY_JITTER = 0.15
DEFAULT_STEP_SUBSTEPS = 1  # align with source attempt_runner (single mj_step per ctrl update)
STABLE_LINEAR = 0.015
STABLE_ANGULAR = 0.15


def food_success_components(
    contacts: list[dict],
    *,
    food_body_id: int,
    fingertip_geom_ids: set[int],
    receptacle_token: str,
    not_dropped: bool,
    food_root_body_id: int | None = None,
    food_body_ids: set[int] | None = None,
) -> dict[str, bool]:
    """Evaluate the explicit release/support/no-drop success contract."""
    matching_body_ids = {int(food_body_id)}
    if food_root_body_id is not None:
        matching_body_ids.add(int(food_root_body_id))
    if food_body_ids is not None:
        matching_body_ids.update(int(body_id) for body_id in food_body_ids)
    food_contacts = [
        pair
        for pair in contacts
        if matching_body_ids.intersection((pair["body1_id"], pair["body2_id"]))
    ]
    released = not any(
        bool({pair["geom1_id"], pair["geom2_id"]} & fingertip_geom_ids) for pair in food_contacts
    )
    plate_supported = any(
        receptacle_token.lower() in pair["body1_name"].lower()
        or receptacle_token.lower() in pair["body2_name"].lower()
        for pair in food_contacts
    )
    not_dropped = bool(not_dropped)
    return {
        "released": bool(released),
        "plate_supported": bool(plate_supported),
        "not_dropped": bool(not_dropped),
        "success": bool(released and plate_supported and not_dropped),
    }


@dataclass
class SamplingBounds:
    """Rectangular XY jitter around each objects nominal home pose."""

    xy_half: float = DEFAULT_XY_JITTER
    z: float = TABLE_Z


class BimanualYamEnv:
    def __init__(
        self,
        *,
        sampling: SamplingBounds | None = None,
        seed: int | None = None,
        step_substeps: int = DEFAULT_STEP_SUBSTEPS,
    ) -> None:
        self.model, self.data, self.arms, self.foods = build_scene()
        self.sampling = sampling or SamplingBounds()
        self._rng = np.random.default_rng(seed)
        self._step_substeps = int(step_substeps)
        self._active_arm: str = "right"
        # capture home qpos for each arm (used to hold the inactive arm still)
        self._home_arm_qpos = {
            side: self.data.qpos[self.arms[side].qaddrs].copy() for side in ("left", "right")
        }
        self._home_finger_qpos = {
            side: self.data.qpos[self.arms[side].finger_qaddrs].copy() for side in ("left", "right")
        }
        # capture nominal object placements (base xyz for jitter) and
        # nominal orientations (as authored by assemble_combined_scene). The
        # model default quats for plate/potato/tomato are non-identity (plate
        # is authored ~90 deg around z; potato/tomato have small tilts). Forcing
        # identity in _set_food_xyz on reset would topple the plate and let it
        # slide off the tabletop.
        self._nominal_food_xyz = self._read_food_xyz()
        self._nominal_food_quat = self._read_food_quat()
        # cached last commanded ctrl so inactive arm keeps its command
        self._last_arm_ctrl = {side: None for side in ("left", "right")}
        self._last_grip_ctrl = {side: OPEN_CTRL for side in ("left", "right")}
        # last IK target per arm; seeds next solve_arm_ik call for continuity
        self._last_ik_target: dict[str, np.ndarray] = {}
        self.action_dim = 7
        # per-arm subtask-signal state (set by external runner during datagen)
        self._arm_signals: dict[str, dict[str, int]] = {"right": {}, "left": {}}
        self._next_reset_offsets: dict[str, np.ndarray] | None = None
        self._last_reset_layout: dict[str, Any] | None = None
        self._post_settle_not_dropped = {"potato": False, "tomato": False}

    # ------------------------------------------------------------------ helpers
    def _read_food_xyz(self) -> dict[str, np.ndarray]:
        return {name: self.data.xpos[body].copy() for name, body in self.foods.items()}

    def _read_food_quat(self) -> dict[str, np.ndarray]:
        """Capture the freejoint quaternion (wxyz) that assemble_combined_scene
        authored for each food body. Used by _set_food_xyz on reset so the
        plate stays plate-side-up."""
        out: dict[str, np.ndarray] = {}
        for name, body in self.foods.items():
            joint_id = int(self.model.body_jntadr[body])
            qposadr = int(self.model.jnt_qposadr[joint_id])
            out[name] = self.data.qpos[qposadr + 3 : qposadr + 7].copy()
        return out

    def _set_food_xyz(self, name: str, xyz: np.ndarray) -> None:
        body = self.foods[name]
        joint_id = int(self.model.body_jntadr[body])
        if joint_id < 0:
            raise RuntimeError(f"food {name} has no free joint")
        qposadr = int(self.model.jnt_qposadr[joint_id])
        # free joint qpos = [x,y,z, qw,qx,qy,qz]
        self.data.qpos[qposadr : qposadr + 3] = xyz
        # Restore the assemble_combined_scene-authored orientation rather than
        # forcing identity. Identity would topple non-trivially-oriented bodies
        # (in particular the plate, which is authored ~90 deg around z).
        self.data.qpos[qposadr + 3 : qposadr + 7] = self._nominal_food_quat[name]
        dofadr = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dofadr : dofadr + 6] = 0.0

    def _apply_ctrl_from_last(self) -> None:
        """Push the cached per-arm ctrl into data.ctrl."""
        for side, arm in self.arms.items():
            if self._last_arm_ctrl[side] is not None:
                self.data.ctrl[arm.actuators] = self._last_arm_ctrl[side]
            for act in arm.gripper:
                self.data.ctrl[act] = self._last_grip_ctrl[side]

    def set_next_reset_offsets(self, offsets: dict[str, np.ndarray]) -> None:
        """Use exact per-object XY offsets on the next reset only."""
        if set(offsets) != set(self.foods):
            raise ValueError(f"offset keys must be {sorted(self.foods)}, got {sorted(offsets)}")
        validated: dict[str, np.ndarray] = {}
        for name, value in offsets.items():
            xy = np.asarray(value, dtype=float).reshape(-1)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                raise ValueError(f"offset for {name} must be a finite XY pair")
            if np.any(np.abs(xy) > self.sampling.xy_half + 1e-12):
                raise ValueError(f"offset for {name} exceeds +/-{self.sampling.xy_half}: {xy}")
            validated[name] = xy.copy()
        self._next_reset_offsets = validated

    def get_last_reset_layout(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_reset_layout)

    # ------------------------------------------------------------------ MimicGen env API
    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        # restore both arms to home pose (from the model default keyframe /
        # first mj_forward in build_scene). build_scene populated qpos already;
        # we re-apply home qpos to be explicit.
        for side, arm in self.arms.items():
            self.data.qpos[arm.qaddrs] = self._home_arm_qpos[side]
            self.data.qpos[arm.finger_qaddrs] = self._home_finger_qpos[side]
            self._last_arm_ctrl[side] = self._home_arm_qpos[side].copy()
            self._last_grip_ctrl[side] = OPEN_CTRL
        # sample new food positions. Preserve each objects nominal z so
        # plate/potato/tomato keep their assemble_combined_scene layering
        # (plate ~1.04, potato ~0.932, tomato ~0.942); a single sampling.z
        # merges all objects into one plane and breaks settle. xy_jitter=0
        # therefore now reproduces the source-demo initial layout exactly.
        explicit_offsets = self._next_reset_offsets
        self._next_reset_offsets = None
        applied_offsets: dict[str, list[float]] = {}
        requested_xyz: dict[str, list[float]] = {}
        for name, xyz0 in self._nominal_food_xyz.items():
            if explicit_offsets is None:
                dx, dy = self._rng.uniform(-self.sampling.xy_half, self.sampling.xy_half, size=2)
            else:
                dx, dy = explicit_offsets[name]
            xyz_new = np.array([xyz0[0] + dx, xyz0[1] + dy, xyz0[2]])
            self._set_food_xyz(name, xyz_new)
            applied_offsets[name] = [float(dx), float(dy)]
            requested_xyz[name] = xyz_new.tolist()
        # let scene settle so objects land on the tabletop. 900 ticks
        # matches the attempt_runner recording warm-up (see
        # yam_food_attempt_runner.py: settle_steps=900). A shorter settle
        # leaves the plate mid-fall when the first ctrl arrives and produces
        # sub-cm drift by end-of-episode, enough to leave tomato hovering.
        self._apply_ctrl_from_last()
        for _ in range(900):
            mujoco.mj_step(self.model, self.data)
        settled_xyz = self._read_food_xyz()
        self._last_reset_layout = {
            "sampling_mode": "explicit" if explicit_offsets is not None else "random",
            "requested_offsets_xy": applied_offsets,
            "nominal_xyz": {name: xyz.tolist() for name, xyz in self._nominal_food_xyz.items()},
            "requested_xyz": requested_xyz,
            "settled_initial_xyz": {name: xyz.tolist() for name, xyz in settled_xyz.items()},
            "settled_relative_xy_to_plate": {
                name: (settled_xyz[name][:2] - settled_xyz["plate"][:2]).tolist()
                for name in ("potato", "tomato")
            },
        }
        # reset per-arm subtask signal state
        self._arm_signals = {"right": {}, "left": {}}
        # clear IK seed cache so first solve_arm_ik after reset uses q_actual
        self._last_ik_target = {}
        return self.get_observation()

    def get_state(self) -> dict[str, Any]:
        return {
            "states": {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "ctrl": self.data.ctrl.copy(),
                "time": float(self.data.time),
            },
            "model_xml": None,  # we always assemble the same XML in-process
        }

    def get_observation(self) -> dict[str, Any]:
        # MimicGen datagen only stores this dict; we do not use it for planning.
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "eef_right_world": self.get_arm_eef_pose_world("right"),
            "eef_left_world": self.get_arm_eef_pose_world("left"),
            "object_poses_world": self.get_object_poses_world(),
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Drive the active arm; hold the inactive arm at its last commanded pose."""
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise RuntimeError(f"action dim {a.shape[0]} != {self.action_dim}")
        arm_targets = a[:6]
        grip = float(a[6])
        active = self.arms[self._active_arm]
        self._last_arm_ctrl[self._active_arm] = arm_targets.copy()
        self._last_grip_ctrl[self._active_arm] = grip
        self._apply_ctrl_from_last()
        for _ in range(self._step_substeps):
            mujoco.mj_step(self.model, self.data)
        # update per-arm subtask signals (populated by external runner via
        # arm_subtask_signals; we do not auto-detect subtask completion here
        # because MimicGen only reads the first 0->1 transition from source)
        obs = self.get_observation()
        return obs, 0.0, False, {}

    def settle_and_measure(self, steps: int = 30) -> None:
        """Run a short settle and require plate support throughout it."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        self._post_settle_not_dropped = {food: True for food in self.foods}
        for _ in range(int(steps)):
            mujoco.mj_step(self.model, self.data)
            contacts = contact_pairs(self.model, self.data)
            for food_name, food_body_id in self.foods.items():
                root_id = int(self.model.body_rootid[food_body_id])
                body_ids = {
                    body_id
                    for body_id in range(self.model.nbody)
                    if int(self.model.body_rootid[body_id]) == root_id
                }
                supported = any(
                    body_ids.intersection((pair["body1_id"], pair["body2_id"]))
                    and (
                        "plate" in pair["body1_name"].lower()
                        or "plate" in pair["body2_name"].lower()
                    )
                    for pair in contacts
                )
                self._post_settle_not_dropped[food_name] &= supported

    def is_success(self) -> dict[str, bool]:
        """Apply release + plate support + post-settle no-drop."""
        contacts = contact_pairs(self.model, self.data)
        reports: dict[str, dict[str, bool]] = {}
        for food_name, side in (("potato", "right"), ("tomato", "left")):
            fingertip_geom_ids = {
                geom_id for geom_ids in self.arms[side].fingertips.values() for geom_id in geom_ids
            }
            food_root_body_id = int(self.model.body_rootid[self.foods[food_name]])
            food_body_ids = {
                body_id
                for body_id in range(self.model.nbody)
                if int(self.model.body_rootid[body_id]) == food_root_body_id
            }
            reports[food_name] = food_success_components(
                contacts,
                food_body_id=self.foods[food_name],
                fingertip_geom_ids=fingertip_geom_ids,
                receptacle_token="plate",
                not_dropped=self._post_settle_not_dropped.get(food_name, False),
                food_root_body_id=food_root_body_id,
                food_body_ids=food_body_ids,
            )

        return {
            "task": bool(reports["potato"]["success"] and reports["tomato"]["success"]),
            "potato_placed": bool(reports["potato"]["success"]),
            "tomato_placed": bool(reports["tomato"]["success"]),
            "potato_released": reports["potato"]["released"],
            "potato_plate_supported": reports["potato"]["plate_supported"],
            "potato_not_dropped": reports["potato"]["not_dropped"],
            "tomato_released": reports["tomato"]["released"],
            "tomato_plate_supported": reports["tomato"]["plate_supported"],
            "tomato_not_dropped": reports["tomato"]["not_dropped"],
        }

    def render(self, mode: str = "rgb_array", **_kw):  # noqa: D401
        raise NotImplementedError("BimanualYamEnv.render is intentionally disabled for datagen.")

    # ------------------------------------------------------------------ EnvInterface hooks
    def set_active_arm(self, side: str) -> None:
        if side not in ("right", "left"):
            raise ValueError(f"side must be right|left, got {side!r}")
        self._active_arm = side

    def get_arm_eef_pose_world(self, side: str) -> np.ndarray:
        arm = self.arms[side]
        # site_xpos already reflects the most recent mj_forward from step().
        pos = self.data.site_xpos[arm.grasp_site].copy()
        rot = self.data.site_xmat[arm.grasp_site].reshape(3, 3).copy()
        H = np.eye(4)
        H[:3, :3] = rot
        H[:3, 3] = pos
        return H

    def fk_arm(self, side: str, arm_qpos: np.ndarray) -> np.ndarray:
        arm = self.arms[side]
        saved = self.data.qpos[arm.qaddrs].copy()
        self.data.qpos[arm.qaddrs] = np.asarray(arm_qpos, dtype=float)
        mujoco.mj_forward(self.model, self.data)
        H = self.get_arm_eef_pose_world(side)
        self.data.qpos[arm.qaddrs] = saved
        mujoco.mj_forward(self.model, self.data)
        return H

    def get_object_poses_world(self) -> dict[str, np.ndarray]:
        # Force object rotation to identity so MimicGens T_new*T_src^-1 becomes
        # a pure translation. Mirrors convert_source_hdf5._obj_hmat. This is
        # compatible with our position-only IK and pick-and-place placement gate.
        poses = {}
        for name, body in self.foods.items():
            H = np.eye(4)
            H[:3, 3] = self.data.xpos[body].copy()
            poses[name] = H
        return poses

    def solve_arm_ik(self, side: str, target_pose_world: np.ndarray) -> np.ndarray:
        """6D IK (position + orientation) via _scene.solve_ik, seeded by the
        previous IK target for temporal continuity across successive calls.

        Rationale: position-only IK let gripper orientation drift, causing
        release-phase failures (tomato leaving the gripper at a wrong angle).
        Seeding from the last IK target (rather than measured qpos) avoids
        the redundant-arm branch jumps that PD tracking lag would otherwise
        induce. Called only from the MimicGen bridge; the attempt runner has
        its own solve_ik at its own call site.
        """
        target = np.asarray(target_pose_world, dtype=float).reshape(4, 4)
        arm = self.arms[side]
        seed = self._last_ik_target.get(side)
        # Position-only IK: matches how the source demo was recorded by
        # attempt_runner.solve_ik. Adding a 6D orientation constraint on top
        # of a source-anchored q seed over-constrains the solver under jitter
        # (source q gives the source wrist orientation, which conflicts with
        # the new position target). Position-only IK with the same source q
        # seed converges on-branch under small perturbations.
        sol, err = solve_ik(self.model, self.data, arm, target[:3, 3], seed=seed)
        # solve_ik now always returns best-effort qpos. Only treat as failure
        # if residual is very large (>2cm), which indicates the target is
        # geometrically unreachable (arm at joint limit / IK diverged).
        if sol is None or err > 0.02:
            # Retry from measured qpos in case seed was pathological
            sol_retry, err_retry = solve_ik(self.model, self.data, arm, target, seed=None)
            if sol_retry is not None and err_retry < err:
                sol, err = sol_retry, err_retry
            if sol is None or err > 0.02:
                raise RuntimeError(
                    f"solve_arm_ik({side}) unreachable target xyz="
                    f"{target[:3, 3].tolist()} err={err:.4f}"
                )
        self._last_ik_target[side] = sol.copy()
        return sol

    def arm_subtask_signals(self, side: str) -> dict[str, int]:
        """Return the current per-arm subtask-signal dict.

        MimicGen only reads the first 0->1 transition to detect subtask
        boundaries in the *source* demo; during generation, waypoint length
        is fully determined by the source subtask segment length. So this
        method is only queried when parsing the source. During generation
        the runner may leave it empty; it exists to satisfy the interface.
        """
        return dict(self._arm_signals.get(side, {}))

    def _set_arm_subtask_signal(self, side: str, name: str, value: int) -> None:
        self._arm_signals.setdefault(side, {})[name] = int(value)

    # ------------------------------------------------------------------ misc
    def clone_seed(self, seed: int) -> "BimanualYamEnv":
        """Return a fresh env with the given RNG seed (for parallel datagen)."""
        return BimanualYamEnv(
            sampling=copy.deepcopy(self.sampling), seed=seed, step_substeps=self._step_substeps
        )
