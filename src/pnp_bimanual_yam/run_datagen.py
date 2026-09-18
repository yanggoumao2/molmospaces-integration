"""Sequential bimanual MimicGen datagen runner for FloorPlan1 YAM pick-and-place.

DexMimicGen-style sequential dispatch, single env.reset() boundary:
    reset() (via right.generate) -> RIGHT arm runs 5 subtasks (potato) ->
    LEFT arm runs 5 subtasks (tomato) on the SAME scene state -> is_success().

Two BimanualYamArmInterface instances share the same BimanualYamEnv. The
left-arm generate() call has env.reset temporarily rebound to a no-op so the
scene state after the right arm persists into the left-arm subtasks. This is
what DexMimicGen means by "sequential" — both arms live inside one physical
episode. Doing two independent gen.generate() calls would re-sample all object
positions between arms, decoupling the arms; that was the pre-Option-A bug.

Source demos come from convert_source_hdf5 (one HDF5 per arm, single demo_0).
Because we only have one source demo, selection_strategy in the per-arm
MG_TaskSpec is forced to "random" (degenerate: always picks demo_0).

Success gate mirrors env_wrapper.is_success(): strict 2/2 stable-and-supported
on the plate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src.pnp_bimanual_yam import _shims as _  # noqa: F401  # register MimicGen shims

# heavyweight imports after shim install
from mimicgen.datagen.data_generator import DataGenerator  # noqa: E402
from mimicgen.utils.file_utils import parse_source_dataset  # noqa: E402

from src.pnp_bimanual_yam.convert_source_hdf5 import convert_arm  # noqa: E402
from src.pnp_bimanual_yam.env_interface import (  # noqa: E402
    BimanualYamArmInterface,
    apply_linear_xy_offset_blend,
)
from src.pnp_bimanual_yam.env_wrapper import BimanualYamEnv, SamplingBounds  # noqa: E402
from src.pnp_bimanual_yam.task_spec import build_task_spec  # noqa: E402


REPO_ROOT = Path(
    os.environ.get("MOLMOSPACES_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()
DEFAULT_SOURCE = Path(
    os.environ.get(
        "YAM_SOURCE_HDF5",
        str(
            REPO_ROOT
            / "work/current/floorplan1_yam_potato_grasp_place_20260831_2038/"
            / "codex_tasks/yam_food_bimanual_20260908_1754/"
            / "execution_attempt_20260914_023902/source_demo_enriched.h5"
        ),
    )
)

RIGHT_CARRY_BLEND = (1915, 2838)
LEFT_CARRY_BLEND = (2513, 3436)


def _apply_attempt_transfer_blend(
    right_gen: DataGenerator,
    left_gen: DataGenerator,
    right_iface: BimanualYamArmInterface,
    left_iface: BimanualYamArmInterface,
    right_base_target_pose: np.ndarray,
    left_base_target_pose: np.ndarray,
    right_base_eefxyz: np.ndarray,
    left_base_eefxyz: np.ndarray,
    layout_offsets: dict[str, np.ndarray] | None,
) -> None:
    """Apply one layout's carry correction without accumulating across attempts."""
    offsets = layout_offsets or {}
    plate = np.asarray(offsets.get("plate", [0.0, 0.0]), dtype=float)
    potato = np.asarray(offsets.get("potato", [0.0, 0.0]), dtype=float)
    tomato = np.asarray(offsets.get("tomato", [0.0, 0.0]), dtype=float)
    right_delta = plate - potato
    left_delta = plate - tomato

    right_gen.src_dataset_infos[0].target_pose = apply_linear_xy_offset_blend(
        right_base_target_pose, *RIGHT_CARRY_BLEND, right_delta
    )
    left_gen.src_dataset_infos[0].target_pose = apply_linear_xy_offset_blend(
        left_base_target_pose, *LEFT_CARRY_BLEND, left_delta
    )

    def blend_positions(base_positions, start, end, delta):
        blended = np.asarray(base_positions, dtype=float).copy()
        alpha = np.linspace(0.0, 1.0, end - start)
        blended[start:end, :2] += alpha[:, None] * delta
        return blended

    right_iface._src_eef_pos = blend_positions(right_base_eefxyz, *RIGHT_CARRY_BLEND, right_delta)
    left_iface._src_eef_pos = blend_positions(left_base_eefxyz, *LEFT_CARRY_BLEND, left_delta)


def _prepare_arm_datagen(
    src_hdf5: Path,
    arm: str,
    workdir: Path,
    *,
    noise: float,
    num_interpolation_steps: int,
) -> tuple[DataGenerator, Path]:
    """Convert source demo (if not cached) and build a DataGenerator for one arm."""
    arm_out = workdir / f"source_{arm}.hdf5"
    if not arm_out.is_file():
        convert_arm(src_hdf5, arm_out, arm=arm)
    spec = build_task_spec(arm, noise=noise, num_interpolation_steps=num_interpolation_steps)
    # single-demo pool: selection_strategy is already random via build_task_spec.
    datagen = DataGenerator(
        task_spec=spec,
        dataset_path=str(arm_out),
        demo_keys=["demo_0"],
    )
    # pre-load the source datagen_info so DataGenerator.generate() can dispatch
    # without re-parsing on every attempt.
    datagen.datagen_infos, datagen.subtask_indices, _, _ = parse_source_dataset(
        dataset_path=str(arm_out),
        demo_keys=["demo_0"],
        task_spec=spec,
    )
    datagen.demo_keys = ["demo_0"]
    return datagen, arm_out


def _run_arm_generate(
    gen: DataGenerator,
    env: BimanualYamEnv,
    iface: BimanualYamArmInterface,
) -> dict:
    """Common kwargs for MG generate() per arm."""
    return gen.generate(
        env=env,
        env_interface=iface,
        select_src_per_subtask=False,
        transform_first_robot_pose=False,
        interpolate_from_last_target_pose=True,
        render=False,
        video_writer=None,
        video_skip=1,
        camera_names=None,
        pause_subtask=False,
    )


def _run_single_attempt(
    env: BimanualYamEnv,
    right_gen: DataGenerator,
    left_gen: DataGenerator,
    right_iface: BimanualYamArmInterface,
    left_iface: BimanualYamArmInterface,
    *,
    seed: int,
    layout_offsets: dict[str, np.ndarray] | None = None,
) -> tuple[dict, dict | None]:
    """Run one bimanual attempt within ONE env.reset() boundary.

    Right arm's generate() performs the sole env.reset() (samples object
    positions). Left arm's generate() then runs against the persisted post-
    right scene state — env.reset is rebound to a no-op for the duration of
    the left-arm generate() call. This matches DexMimicGen sequential mode:
    both arms operate on the same physical episode.
    """
    t0 = time.time()
    np.random.seed(seed)
    env._rng = np.random.default_rng(seed)  # deterministic per attempt
    env._last_reset_layout = None
    if layout_offsets is not None:
        env.set_next_reset_offsets(layout_offsets)
    # NOTE: do NOT env.reset() here. MG's DataGenerator.generate() calls
    # env.reset() internally at the start of every generate() call. If we
    # also reset here, our IK seed / passthrough logic references reset #1's
    # sampling, but MG's transformed target uses reset #2's sampling.
    trace: dict = {"seed": int(seed), "arms": {}}

    # ---------------- RIGHT arm (owns the env.reset) ----------------
    right_iface.reset_step_counter()
    env.set_active_arm("right")
    try:
        r_result = _run_arm_generate(right_gen, env, right_iface)
    except Exception as exc:
        trace["arms"]["right"] = {
            "status": "exception",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        # Even if right arm blew up, we still record success_report from
        # whatever scene state exists — but we skip the left arm because
        # its generate() would call env.reset() (bypassed below) or produce
        # garbage on a corrupted state.
        trace["success_report"] = {k: bool(v) for k, v in env.is_success().items()}
        trace["success"] = False
        trace["elapsed_s"] = time.time() - t0
        trace["initial_layout"] = env.get_last_reset_layout()
        trace["final_xyz"] = _food_xyz(env) if trace["initial_layout"] is not None else None
        return trace, None
    trace["arms"]["right"] = {
        "status": "ok",
        "num_steps": int(r_result["actions"].shape[0])
        if hasattr(r_result["actions"], "shape")
        else 0,
        "arm_success_report": bool(r_result.get("success", False)),
    }

    # ---------------- LEFT arm (reset temporarily disabled) ----------------
    # DexMimicGen sequential contract: left arm runs on the SAME scene state
    # produced by the right arm. MG generate() unconditionally calls
    # env.reset() at its top; we neutralize it just for this call.
    left_iface.reset_step_counter()
    env.set_active_arm("left")
    _orig_reset = env.reset
    env.reset = lambda *a, **k: None  # noqa: E731  # scoped to this generate() call
    try:
        try:
            l_result = _run_arm_generate(left_gen, env, left_iface)
        except Exception as exc:
            trace["arms"]["left"] = {
                "status": "exception",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            trace["success_report"] = {k: bool(v) for k, v in env.is_success().items()}
            trace["success"] = False
            trace["elapsed_s"] = time.time() - t0
            trace["initial_layout"] = env.get_last_reset_layout()
            trace["final_xyz"] = _food_xyz(env) if trace["initial_layout"] is not None else None
            return trace, None
    finally:
        env.reset = _orig_reset

    trace["arms"]["left"] = {
        "status": "ok",
        "num_steps": int(l_result["actions"].shape[0])
        if hasattr(l_result["actions"], "shape")
        else 0,
        "arm_success_report": bool(l_result.get("success", False)),
    }

    # final scene-level success gate after a bounded settle window
    env.settle_and_measure()
    success_report = env.is_success()
    trace["success_report"] = {k: bool(v) for k, v in success_report.items()}
    trace["success"] = bool(success_report.get("task", False))
    trace["elapsed_s"] = time.time() - t0
    trace["initial_layout"] = env.get_last_reset_layout()
    trace["final_xyz"] = _food_xyz(env)
    return trace, {"right": r_result, "left": l_result}


def _food_xyz(env: BimanualYamEnv) -> dict[str, list[float]]:
    return {name: env.data.xpos[body].astype(float).tolist() for name, body in env.foods.items()}


def _latin_hypercube_layouts(
    count: int,
    xy_half: float,
    seed: int,
) -> list[dict]:
    """Build a deterministic 6-D Latin-hypercube over three object XY pairs."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    dimensions = [(name, axis) for name in ("potato", "tomato", "plate") for axis in ("x", "y")]
    unit = np.empty((count, len(dimensions)), dtype=float)
    strata = np.empty((count, len(dimensions)), dtype=int)
    for dim in range(len(dimensions)):
        permutation = rng.permutation(count)
        strata[:, dim] = permutation
        unit[:, dim] = (permutation + rng.random(count)) / count
    scaled = (2.0 * unit - 1.0) * float(xy_half)
    layouts = []
    for sample_index in range(count):
        offsets = {}
        stratum_by_coordinate = {}
        for dim, (name, axis) in enumerate(dimensions):
            offsets.setdefault(name, [0.0, 0.0])[0 if axis == "x" else 1] = float(
                scaled[sample_index, dim]
            )
            stratum_by_coordinate[f"{name}_{axis}"] = int(strata[sample_index, dim])
        layouts.append(
            {
                "sample_index": sample_index,
                "offsets_xy": offsets,
                "stratum_by_coordinate": stratum_by_coordinate,
            }
        )
    return layouts


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_path, path)


def _save_success_trajectory(path: Path, results: dict) -> None:
    """Store compact per-arm arrays for strict-success pilot inspection."""

    def stack_state(arm: str, field: str) -> np.ndarray:
        values = [state[field] for state in results[arm]["states"]]
        return np.asarray(values)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        right_actions=np.asarray(results["right"]["actions"]),
        right_qpos=stack_state("right", "qpos"),
        right_qvel=stack_state("right", "qvel"),
        right_ctrl=stack_state("right", "ctrl"),
        right_time=stack_state("right", "time"),
        left_actions=np.asarray(results["left"]["actions"]),
        left_qpos=stack_state("left", "qpos"),
        left_qvel=stack_state("left", "qvel"),
        left_ctrl=stack_state("left", "ctrl"),
        left_time=stack_state("left", "time"),
    )
    os.replace(tmp_path, path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src", type=Path, default=DEFAULT_SOURCE, help="path to source_demo_enriched.h5"
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=Path("runtime/mimicgen_pick_and_place/bimanual_v1"),
        help="working directory (converted sources + attempt logs)",
    )
    p.add_argument(
        "--num-attempts",
        type=int,
        default=10,
        help="number of bimanual attempts to run (smoke default: 10)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sampling-design",
        choices=("random", "lhs"),
        default="random",
        help="random per-attempt reset or deterministic Latin hypercube",
    )
    p.add_argument("--xy-jitter", type=float, default=0.15)
    p.add_argument("--table-z", type=float, default=0.9323)
    p.add_argument("--step-substeps", type=int, default=1)
    p.add_argument("--action-noise", type=float, default=0.0)
    p.add_argument("--num-interpolation-steps", type=int, default=5)
    p.add_argument(
        "--manifest", type=Path, default=None, help="write per-attempt summary JSON to this path"
    )
    p.add_argument(
        "--resume", action="store_true", help="resume a compatible checkpointed manifest"
    )
    p.add_argument(
        "--success-dir",
        type=Path,
        default=None,
        help="save compact NPZ only for strict-success attempts",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    src = Path(args.src).resolve() if args.src.is_absolute() else Path.cwd() / args.src
    if not src.is_file():
        print(f"ERROR: source demo not found at {src}", file=sys.stderr)
        return 2
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[bimanual-datagen] source={src}")
    print(f"[bimanual-datagen] workdir={workdir}")

    right_gen, right_src = _prepare_arm_datagen(
        src,
        "right",
        workdir,
        noise=args.action_noise,
        num_interpolation_steps=args.num_interpolation_steps,
    )
    left_gen, left_src = _prepare_arm_datagen(
        src,
        "left",
        workdir,
        noise=args.action_noise,
        num_interpolation_steps=args.num_interpolation_steps,
    )
    print(f"[bimanual-datagen] right_src={right_src} left_src={left_src}")

    env = BimanualYamEnv(
        sampling=SamplingBounds(xy_half=args.xy_jitter, z=args.table_z),
        seed=args.seed,
        step_substeps=args.step_substeps,
    )
    # Load per-arm source qpos + eef xyz sequences to enable nearest-neighbor
    # source-seeded IK inside target_pose_to_action. Under identity smoke this
    # collapses IK to q_src exactly; under jitter it keeps IK on the source
    # solution branch (avoiding 6-DoF branch jumps).
    import h5py as _h5

    def _load_src_seed(hpath):
        with _h5.File(hpath, "r") as f:
            acts = f["data/demo_0/actions"][:]  # (T, 7)
            eefp = f["data/demo_0/datagen_info/target_pose"][
                :
            ]  # (T, 4, 4) — commanded, not achieved
        return acts[:, :6].astype(float), eefp[:, :3, 3].astype(float)

    r_qseq, r_eefxyz = _load_src_seed(right_src)
    l_qseq, l_eefxyz = _load_src_seed(left_src)
    right_base_target_pose = np.asarray(
        right_gen.src_dataset_infos[0].target_pose, dtype=float
    ).copy()
    left_base_target_pose = np.asarray(
        left_gen.src_dataset_infos[0].target_pose, dtype=float
    ).copy()
    right_base_eefxyz = r_eefxyz.copy()
    left_base_eefxyz = l_eefxyz.copy()
    right_iface = BimanualYamArmInterface(
        env, "right", source_qpos_seq=r_qseq, source_eef_pos_seq=r_eefxyz
    )
    left_iface = BimanualYamArmInterface(
        env, "left", source_qpos_seq=l_qseq, source_eef_pos_seq=l_eefxyz
    )

    manifest_path = args.manifest or (workdir / "smoke_manifest.json")
    layouts = (
        _latin_hypercube_layouts(args.num_attempts, args.xy_jitter, args.seed)
        if args.sampling_design == "lhs"
        else [{"sample_index": i, "offsets_xy": None} for i in range(args.num_attempts)]
    )
    manifest_config = {
        "src": str(src),
        "workdir": str(workdir),
        "num_attempts": int(args.num_attempts),
        "seed": int(args.seed),
        "sampling_design": args.sampling_design,
        "xy_jitter": float(args.xy_jitter),
        "table_z": float(args.table_z),
        "step_substeps": int(args.step_substeps),
        "action_noise": float(args.action_noise),
    }
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        previous_config = {key: manifest.get(key) for key in manifest_config}
        if previous_config != manifest_config:
            raise RuntimeError(
                f"resume config mismatch: previous={previous_config}, current={manifest_config}"
            )
    else:
        if manifest_path.exists():
            raise FileExistsError(
                f"manifest already exists: {manifest_path}; pass --resume or choose another path"
            )
        manifest = {
            "schema_version": 2,
            **manifest_config,
            "design": layouts,
            "attempts": [],
            "n_completed": 0,
            "n_success": 0,
            "hit_rate": 0.0,
        }
        _atomic_write_json(manifest_path, manifest)

    completed = {int(item["sample_index"]) for item in manifest["attempts"]}
    for i, layout in enumerate(layouts):
        if i in completed:
            continue
        seed_i = int(args.seed) + i
        offsets = layout.get("offsets_xy")
        np_offsets = (
            None
            if offsets is None
            else {name: np.asarray(value, dtype=float) for name, value in offsets.items()}
        )
        _apply_attempt_transfer_blend(
            right_gen,
            left_gen,
            right_iface,
            left_iface,
            right_base_target_pose,
            left_base_target_pose,
            right_base_eefxyz,
            left_base_eefxyz,
            np_offsets,
        )
        trace, results = _run_single_attempt(
            env,
            right_gen,
            left_gen,
            right_iface,
            left_iface,
            seed=seed_i,
            layout_offsets=np_offsets,
        )
        trace["sample_index"] = i
        trace["design_offsets_xy"] = offsets
        trace["stratum_by_coordinate"] = layout.get("stratum_by_coordinate")
        if trace.get("success") and args.success_dir is not None and results is not None:
            artifact = args.success_dir / f"sample_{i:03d}_seed_{seed_i}.npz"
            _save_success_trajectory(artifact, results)
            trace["success_artifact"] = str(artifact)
        manifest["attempts"].append(trace)
        manifest["attempts"].sort(key=lambda item: int(item["sample_index"]))
        manifest["n_completed"] = len(manifest["attempts"])
        manifest["n_success"] = sum(bool(item.get("success")) for item in manifest["attempts"])
        manifest["hit_rate"] = manifest["n_success"] / max(1, manifest["n_completed"])
        _atomic_write_json(manifest_path, manifest)
        print(
            f"[bimanual-datagen] attempt {i + 1}/{args.num_attempts} "
            f"seed={seed_i} success={trace.get('success')} "
            f"elapsed={trace.get('elapsed_s'):.2f}s",
            flush=True,
        )
    print(f"[bimanual-datagen] wrote {manifest_path}")
    print(
        f"[bimanual-datagen] n_success={manifest['n_success']}/{manifest['n_completed']} "
        f"hit_rate={manifest['hit_rate']:.2%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
