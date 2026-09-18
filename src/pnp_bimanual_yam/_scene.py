"""Scene assembly + entity adapters for FloorPlan1 bimanual YAM datagen.

Ported from work/.../yam_food_attempt_runner.py.pre_source_demo_20260911. Removed
diagnostic/reporting code (frame capture, MP4, stability audit, IK diagnostics).
Kept scene assembly, arm/food id resolution, IK, and support-contact checks.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(
    os.environ.get("MOLMOSPACES_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()
TASK_ROOT = Path(
    os.environ.get(
        "YAM_FOOD_TASK_DIR",
        str(
            REPO_ROOT / "work/current/floorplan1_yam_potato_grasp_place_20260831_2038/"
            "codex_tasks/yam_food_grasp_place_20260906_1109"
        ),
    )
).resolve()
sys.path.insert(0, str(TASK_ROOT))

from correction_assembly.assemble_combined_scene import (  # noqa: E402
    FOOD_POSITIONS,
    REQUESTED_YAM_XML,
    assemble_model,
)


FOOD_TOKENS = {"plate": "plate_", "potato": "Irishpotato_", "tomato": "tomato_"}
OPEN_CTRL, CLOSE_CTRL = -0.0475, 0.0


def body_id(model: mujoco.MjModel, token: str) -> int:
    for index in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or ""
        if token.lower() in name.lower():
            return index
    raise KeyError(token)


def contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    pairs: list[dict] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        b1 = int(model.geom_bodyid[contact.geom1])
        b2 = int(model.geom_bodyid[contact.geom2])
        pairs.append(
            {
                "geom1_id": int(contact.geom1),
                "geom2_id": int(contact.geom2),
                "body1_id": b1,
                "body2_id": b2,
                "body1_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or "",
                "body2_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or "",
            }
        )
    return pairs


@dataclass(frozen=True)
class ArmAdapter:
    side: str
    joints: list[int]
    actuators: list[int]
    gripper: list[int]
    grasp_site: int
    fingertips: dict[str, list[int]]
    qaddrs: list[int]
    daddrs: list[int]
    finger_qaddrs: list[int]


def arm_ids(model: mujoco.MjModel, side: str) -> ArmAdapter:
    side = side.lower()
    joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_0/{side}_joint{i}")
        for i in range(1, 7)
    ]
    actuators = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot_0/{side}_joint{i}")
        for i in range(1, 7)
    ]
    gripper = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot_0/{side}_gripper_{tip}")
        for tip in ("left_tip", "right_tip")
    ]
    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot_0/{side}_grasp_site")
    fingertips = {
        "left": [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"robot_0/{side}_left_tip_collider_{i}"
            )
            for i in range(3)
        ],
        "right": [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"robot_0/{side}_right_tip_collider_{i}"
            )
            for i in range(3)
        ],
    }
    if (
        min(*joints, *actuators, *gripper, grasp_site, *fingertips["left"], *fingertips["right"])
        < 0
    ):
        raise RuntimeError(f"required {side}-arm entity absent")
    return ArmAdapter(
        side=side,
        joints=joints,
        actuators=actuators,
        gripper=gripper,
        grasp_site=grasp_site,
        fingertips=fingertips,
        qaddrs=[int(model.jnt_qposadr[j]) for j in joints],
        daddrs=[int(model.jnt_dofadr[j]) for j in joints],
        finger_qaddrs=[int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])]) for a in gripper],
    )


def build_scene() -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, ArmAdapter], dict[str, int]]:
    """Assemble the FloorPlan1 bimanual YAM + food scene and resolve entity ids.

    Returns (model, data, arms, foods) where arms maps side->ArmAdapter and
    foods maps label->body id."""
    _spec, model, data, _qpos_addresses = assemble_model(REQUESTED_YAM_XML)
    mujoco.mj_forward(model, data)
    arms = {side: arm_ids(model, side) for side in ("left", "right")}
    foods = {label: body_id(model, token) for label, token in FOOD_TOKENS.items()}
    return model, data, arms, foods


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmAdapter,
    target,
    max_iter: int = 420,
    tol: float = 0.003,
    tol_rot: float = 0.05,
    damping: float = 2e-3,
    step_size: float = 0.22,
    seed: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Jacobian IK for arm.grasp_site.

    target: either (3,) world xyz for position-only IK, or (4,4) world pose
        for 6D IK (position + orientation). 6D form is used by the MimicGen
        bridge so end-effector orientation follows the source demo.
    seed: optional (n_joints,) initial qpos for this arm. When provided, IK
        starts from  instead of the currently measured data.qpos. This
        gives temporal continuity across successive IK calls (each call seeds
        from the previous IK target) rather than from lagging PD-tracked
        state, avoiding branch jumps in the redundant 6-DoF arm.
    tol / tol_rot: convergence thresholds for position (m) and rotation (rad).
        Default tol tightened from 4mm to 1mm to reduce per-step residual that
        would otherwise accumulate over thousands of steps.

    Restores qpos on both success and failure so callers see no state leak."""
    joints = arm.joints
    qaddrs = arm.qaddrs
    daddrs = arm.daddrs
    site = arm.grasp_site
    original = data.qpos[qaddrs].copy()

    target_arr = np.asarray(target, dtype=float)
    if target_arr.shape == (4, 4):
        target_xyz = target_arr[:3, 3]
        target_R = target_arr[:3, :3]
        use_orient = True
    else:
        target_xyz = target_arr.reshape(3)
        target_R = None
        use_orient = False

    if seed is not None:
        seed_arr = np.asarray(seed, dtype=float).reshape(-1)
        if seed_arr.shape[0] == len(qaddrs):
            for qaddr, sv, joint in zip(qaddrs, seed_arr, joints, strict=True):
                data.qpos[qaddr] = np.clip(sv, *model.jnt_range[joint])

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        pos_err = target_xyz - data.site_xpos[site]
        pos_norm = float(np.linalg.norm(pos_err))

        if use_orient:
            R_cur = data.site_xmat[site].reshape(3, 3)
            R_diff = target_R @ R_cur.T
            cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle = float(np.arccos(cos_a))
            if angle < 1e-8:
                rot_err = np.zeros(3)
                rot_norm = 0.0
            else:
                axis = np.array(
                    [
                        R_diff[2, 1] - R_diff[1, 2],
                        R_diff[0, 2] - R_diff[2, 0],
                        R_diff[1, 0] - R_diff[0, 1],
                    ]
                ) / (2.0 * np.sin(angle))
                rot_err = angle * axis
                rot_norm = float(np.linalg.norm(rot_err))
            if pos_norm < tol and rot_norm < tol_rot:
                result = data.qpos[qaddrs].copy()
                data.qpos[qaddrs] = original
                mujoco.mj_forward(model, data)
                return result, pos_norm
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            J = np.vstack([jp[:, daddrs], jr[:, daddrs]])
            err6 = np.concatenate([pos_err, rot_err])
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err6)
        else:
            if pos_norm < tol:
                result = data.qpos[qaddrs].copy()
                data.qpos[qaddrs] = original
                mujoco.mj_forward(model, data)
                return result, pos_norm
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            J = jp[:, daddrs]
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), pos_err)

        for joint, qaddr, delta in zip(joints, qaddrs, step, strict=True):
            data.qpos[qaddr] = np.clip(
                data.qpos[qaddr] + step_size * delta, *model.jnt_range[joint]
            )

    # Return best-effort result rather than None: MimicGen tolerates small
    # per-step residual, so raising here on identity-close targets would abort
    # trajectories that are still trackable. Only when the residual is very
    # large (>2cm) does the caller treat it as a real failure.
    mujoco.mj_forward(model, data)
    err_final = float(np.linalg.norm(target_xyz - data.site_xpos[site]))
    best_q = data.qpos[qaddrs].copy()
    data.qpos[qaddrs] = original
    mujoco.mj_forward(model, data)
    return best_q, err_final


def support_contacts_for_food(
    model: mujoco.MjModel, data: mujoco.MjData, food_body: int
) -> list[dict]:
    """Return contacts under a food body from non-robot supports."""
    food_root = int(model.body_rootid[food_body])
    supports: list[dict] = []
    for pair in contact_pairs(model, data):
        if int(model.body_rootid[pair["body1_id"]]) == food_root:
            support_name = pair["body2_name"]
        elif int(model.body_rootid[pair["body2_id"]]) == food_root:
            support_name = pair["body1_name"]
        else:
            continue
        if support_name.startswith("robot_0/"):
            continue
        supports.append({"support_body": support_name})
    return supports


def food_stable_and_supported_by(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    food_body: int,
    receptacle_token: str,
    *,
    linear_v_thresh: float = 0.015,
    angular_v_thresh: float = 0.15,
) -> tuple[bool, dict]:
    """Return (ok, diag) for `food is stable and supported by any body whose
    name contains receptacle_token`."""
    joint_id = int(model.body_jntadr[food_body])
    if joint_id < 0:
        return False, {"reason": "food has no joint"}
    dofadr = int(model.jnt_dofadr[joint_id])
    lin_v = np.linalg.norm(data.qvel[dofadr : dofadr + 3])
    ang_v = np.linalg.norm(data.qvel[dofadr + 3 : dofadr + 6])
    stable = bool(lin_v < linear_v_thresh and ang_v < angular_v_thresh)
    supports = support_contacts_for_food(model, data, food_body)
    supported = any(receptacle_token.lower() in s["support_body"].lower() for s in supports)
    return stable and supported, {
        "linear_velocity_norm": float(lin_v),
        "angular_velocity_norm": float(ang_v),
        "stable": stable,
        "supports": supports,
        "supported_by_receptacle": supported,
    }
