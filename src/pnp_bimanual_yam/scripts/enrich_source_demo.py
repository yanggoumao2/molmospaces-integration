#!/usr/bin/env python3
"""Offline replay source_demo.hdf5 to recover full 6-DOF object and EE poses.

For each recorded qpos_full frame, mj_forward and read:
  - free-object body xpos/xquat for plate/potato/tomato
  - grasp-site xpos + xmat (3x3) for left and right arms

Writes source_demo_enriched.h5 alongside the original.
"""

from __future__ import annotations
import os, sys, time
from pathlib import Path
import h5py
import mujoco
import numpy as np

REPO = Path(os.environ.get("MOLMOSPACES_ROOT", str(Path(__file__).resolve().parents[3]))).resolve()
TASK_DIR = Path(
    os.environ.get(
        "YAM_FOOD_TASK_DIR",
        str(
            REPO
            / "work/current/floorplan1_yam_potato_grasp_place_20260831_2038/"
            / "codex_tasks/yam_food_bimanual_20260908_1754"
        ),
    )
).resolve()
ATTEMPT = Path(
    os.environ.get(
        "YAM_SOURCE_ATTEMPT",
        TASK_DIR / "execution_attempt_20260911_121406",
    )
)
SRC_H5 = Path(os.environ.get("YAM_SOURCE_HDF5", ATTEMPT / "source_demo.hdf5"))
OUT_H5 = Path(
    os.environ.get(
        "YAM_ENRICHED_HDF5",
        ATTEMPT / "source_demo_enriched.h5",
    )
)

sys.path.insert(0, str(TASK_DIR / "correction_assembly"))
import assemble_combined_scene as ACS

t0 = time.time()
scene_spec, model, data, ids = ACS.assemble_model(ACS.REQUESTED_YAM_XML)
print(f"[{time.time() - t0:.1f}s] Assembled model: nq={model.nq}, nu={model.nu}")


# --- Body / site lookups ---
def bid(name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def sid(name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))


food_bodies = {}
for label, tok in ACS.FOOD_TOKENS.items():
    for i in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n and tok.lower() in n.lower() and model.body_jntadr[i] >= 0:
            food_bodies[label] = (n, i)
            break
print("Food bodies:", {k: v[0] for k, v in food_bodies.items()})

left_site = sid("robot_0/left_grasp_site")
right_site = sid("robot_0/right_grasp_site")
assert left_site >= 0 and right_site >= 0, "grasp sites missing"

# --- Read source demo qpos_full and replay ---
with h5py.File(SRC_H5, "r") as f:
    src = f["traj_0/mimicgen_yam"]
    qpos_full = src["qpos_full"][:]
    time_s = src["time_s"][:]
    left_arm = src["arm_qpos/left"][:]
    right_arm = src["arm_qpos/right"][:]
    left_fing = src["arm_qpos/left_finger"][:]
    right_fing = src["arm_qpos/right_finger"][:]
    lg_cmd = src["replay_commands/left_gripper"][:]
    rg_cmd = src["replay_commands/right_gripper"][:]
    la_cmd = src["replay_commands/left_arm"][:]
    ra_cmd = src["replay_commands/right_arm"][:]
    src_traj_attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in src.attrs.items()}
    src_object_xyz = {k: src[f"object_pose/{k}"][:] for k in ("plate", "potato", "tomato")}
    src_tcp_left = src["left_tcp_world_xyz"][:]
    src_tcp_right = src["right_tcp_world_xyz"][:]
    scene_constraints_json = src["scene_constraints_json"][()]
    gates_json = src["gates_json"][()]
N = qpos_full.shape[0]
print(f"[{time.time() - t0:.1f}s] Loaded source demo: N={N} steps, nq={qpos_full.shape[1]}")
assert qpos_full.shape[1] == model.nq, (qpos_full.shape, model.nq)

# --- Replay loop ---
obj_xpos = {k: np.zeros((N, 3), np.float32) for k in ("plate", "potato", "tomato")}
obj_xquat = {k: np.zeros((N, 4), np.float32) for k in ("plate", "potato", "tomato")}
left_ee_xpos = np.zeros((N, 3), np.float32)
right_ee_xpos = np.zeros((N, 3), np.float32)
left_ee_xmat = np.zeros((N, 9), np.float32)  # row-major 3x3
right_ee_xmat = np.zeros((N, 9), np.float32)

for t in range(N):
    data.qpos[:] = qpos_full[t]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for label, (_, body_id) in food_bodies.items():
        obj_xpos[label][t] = data.xpos[body_id]
        obj_xquat[label][t] = data.xquat[body_id]
    left_ee_xpos[t] = data.site_xpos[left_site]
    right_ee_xpos[t] = data.site_xpos[right_site]
    left_ee_xmat[t] = data.site_xmat[left_site]
    right_ee_xmat[t] = data.site_xmat[right_site]
    if t % 2000 == 0:
        print(f"[{time.time() - t0:.1f}s]  replayed {t}/{N}")

# --- Sanity: xyz recovered from replay should match stored xyz within eps ---
for label in ("plate", "potato", "tomato"):
    d = np.linalg.norm(obj_xpos[label] - src_object_xyz[label], axis=1)
    print(f"  {label} xyz recovery max|delta| = {d.max():.2e} m (mean {d.mean():.2e})")
d_lt = np.linalg.norm(left_ee_xpos - src_tcp_left, axis=1)
d_rt = np.linalg.norm(right_ee_xpos - src_tcp_right, axis=1)
print(f"  left EE xyz  max|delta| = {d_lt.max():.2e} m")
print(f"  right EE xyz max|delta| = {d_rt.max():.2e} m")

# --- Write enriched HDF5 ---
with h5py.File(OUT_H5, "w") as f:
    g = f.create_group("traj_0")
    g.attrs["schema_version"] = "0.4-enriched"
    g.attrs["source_hdf5"] = str(SRC_H5)
    for k, v in src_traj_attrs.items():
        g.attrs[f"orig_{k}"] = v
    m = g.create_group("mimicgen_yam")
    # Time / qpos
    m.create_dataset("time_s", data=time_s)
    m.create_dataset("qpos_full", data=qpos_full, compression="lzf")
    # Object poses (full 6-DOF)
    op = m.create_group("object_pose")
    for label in ("plate", "potato", "tomato"):
        og = op.create_group(label)
        og.create_dataset("pos", data=obj_xpos[label], compression="lzf")
        og.create_dataset("quat_wxyz", data=obj_xquat[label], compression="lzf")
        og.attrs["body_name"] = food_bodies[label][0]
    # EE poses (full 6-DOF, xmat is 3x3 row-major)
    ee = m.create_group("ee_pose")
    for arm, xp, xm in (
        ("left", left_ee_xpos, left_ee_xmat),
        ("right", right_ee_xpos, right_ee_xmat),
    ):
        eg = ee.create_group(arm)
        eg.create_dataset("pos", data=xp, compression="lzf")
        eg.create_dataset("rot_mat_row_major", data=xm.astype(np.float32), compression="lzf")
        eg.attrs["site_name"] = f"robot_0/{arm}_grasp_site"
    # Arm + gripper qpos and commanded actions (preserve verbatim)
    aq = m.create_group("arm_qpos")
    aq.create_dataset("left", data=left_arm, compression="lzf")
    aq.create_dataset("right", data=right_arm, compression="lzf")
    aq.create_dataset("left_finger", data=left_fing, compression="lzf")
    aq.create_dataset("right_finger", data=right_fing, compression="lzf")
    rc = m.create_group("replay_commands")
    rc.create_dataset("left_arm", data=la_cmd, compression="lzf")
    rc.create_dataset("right_arm", data=ra_cmd, compression="lzf")
    rc.create_dataset("left_gripper", data=lg_cmd, compression="lzf")
    rc.create_dataset("right_gripper", data=rg_cmd, compression="lzf")
    # Preserve pass-through metadata
    m.create_dataset("scene_constraints_json", data=scene_constraints_json)
    m.create_dataset("gates_json", data=gates_json)

print(f"[{time.time() - t0:.1f}s] Wrote {OUT_H5}, size = {OUT_H5.stat().st_size} bytes")
