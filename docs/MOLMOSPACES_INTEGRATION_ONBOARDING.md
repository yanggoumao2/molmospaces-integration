# MolmoSpaces Integration -- Developer Onboarding Guide

**Audit snapshot:** 2026-09-08 CST
**Audited checkout:** the integration repository checkout
**Branch / base commit at audit:** `main` / `c64d3ba`
**Audit scope:** static, read-only architecture and integration review. No simulation, generation, training, evaluation, or source changes were started by this audit.

## 1. Overview

This repository combines the upstream MolmoSpaces MuJoCo data-generation framework with two local integration worklines:

- `src/pnp/`: a MimicGen-compatible fixed-base Franka pick-and-place adapter and controlled rollout-generation workflow.
- `src/bimanual_yam/`: a separate bimanual YAM browser-teleoperation, source-demo, and iTHOR diagnostic workflow.

The boundary is important. Upstream MolmoSpaces owns experiment configuration, scene/task sampling, MuJoCo lifecycle, robot controllers, policies, sensors, trajectory serialization, and generic data-generation orchestration. The PnP and YAM directories add project-specific embodiments, execution bridges, provenance, and evidence gates; they are not upstream framework guarantees.

The checkout had existing, uncommitted changes at audit time. They were examined but never modified or reverted. Current git state, dependency availability, assets, renderer support, and generated artifacts are mutable and must be rechecked before running work.

## 2. Quick Orientation

### Runtime and entry points

- Primary language/runtime: Python package defined by `pyproject.toml`.
- Primary upstream data-generation entry: `molmo_spaces/data_generation/main.py`.
- Generic execution shape: `python -m molmo_spaces.data_generation.main <RegisteredConfigName>`.
- Pipeline orchestration: `molmo_spaces/data_generation/pipeline.py` and `ParallelRolloutRunner`.
- Configuration registry: `molmo_spaces/data_generation/config_registry.py`.
- PnP batch workflow: `src/pnp/run_generation.py`.
- PnP single-rollout bridge: `src/pnp/generate_pick_place_rollout.py`.
- YAM browser source-demo path: `src/bimanual_yam/browser_keyboard_teleop.py`.
- YAM scripted diagnostic path: `src/bimanual_yam/scripted_bimanual_source_demo.py`.

### Environment variables that affect runtime behavior

| Variable | Role |
|---|---|
| `MLSPACES_ASSETS_DIR` | MolmoSpaces resource and scene/robot asset location. |
| `MLSPACES_CACHE_DIR` | Cache location used by the framework. |
| `HF_HOME` | Hugging Face cache location. |
| `NLTK_DATA`, `MOLMOSPACES_NLTK_DATA` | NLTK resource lookup paths. |
| `MUJOCO_GL`, `PYOPENGL_PLATFORM` | Renderer backend selection. Treat as host-specific. |
| `PYTHONPATH` | Required when using local PnP/YAM sources or vendored dependencies. |

Do not treat a successful import or EGL context creation as behavioral success. A real camera frame, valid task initialization, command execution, task predicate, and replay gate are distinct layers.

## 3. Architecture

```text
                         registered experiment config
                                     |
                                     v
     data_generation/main.py --> config_registry --> ParallelRolloutRunner
                                                      |
                                                      v
                                       BaseMujocoTaskSampler.sample_task()
                                      /                              \
                      scene load/reuse + object placement       robot / policy setup
                                      |                              |
                                      +--------------+---------------+
                                                     v
                                         BaseMujocoTask.reset()
                                                     |
                        sensor observation <--------+--------> policy / teleop action
                                                     |
                                                     v
                                         BaseMujocoTask.step(action)
                                      robot.update_control(action)
                                      -> robot.compute_control()
                                      -> CPUMujocoEnv.step()
                                      -> sensors/reward/success/history
                                                     |
                                                     v
                                               HDF5 trajectory

PnP path:
source HDF5 + target manifest --> run_generation.py --> generate_pick_place_rollout.py
--> MolmoSpacesPnpEnv / MG_MolmoSpacesPickAndPlace --> MimicGen waypoint execution
--> artifacts, HDF5, video, result JSON, provenance JSONL

YAM path:
browser_keyboard_teleop.py or scripted_bimanual_source_demo.py
--> bimanual YAM actions --> MolmoSpaces task history/HDF5
--> strict capture metadata + replay commands + fresh-reset replay evidence
```

### Upstream lifecycle and state/action boundaries

`BaseMujocoTaskSampler.sample_task()` advances/selects a house, loads a scene only when the house changes, otherwise restores a scene-preset task configuration, then prepares the task-specific scene state. The sampler owns the long-lived environment and is responsible for scene reuse; tasks should not independently close that environment.

`BaseMujocoTask.reset()` clears episode counters and cached histories, resets stateful sensors and registered policy, obtains the initial observation, and freezes the sampled configuration. The reset observation is part of the trajectory contract.

`BaseMujocoTask.step(action)` accepts either one action dict for `n_batch=1` or one dict per batch element. It applies each action through `robot.update_control`, executes configured control and physics substeps, polls sensors, computes reward/terminal/truncation/success information, and appends action/history state. A literal `{"done": true}` is consumed as task termination intent.

`CPUMujocoEnv` owns MuJoCo model/data lifecycle. `BaseMujocoTask.close()` drops task-level references but deliberately does not close the sampler-owned environment.

For `PickAndPlaceTask`, reward is a dense position measure, while success is a stronger task predicate using object/receptacle state and support/contact logic. Do not report low position error, IK convergence, or a valid HDF5 as pick-and-place success.

## 4. Key Concepts

### Config -> sampler -> task -> environment -> HDF5

1. A registered config selects task, sampler, robot, cameras, policy, output, and rollout limits.
2. A task sampler picks/loads a house and samples concrete object, receptacle, robot, and scene state.
3. A task wraps the sampler-owned MuJoCo environment with sensors, task descriptions, reward, termination, success predicates, and trajectory history.
4. Robot actions are controller-specific dicts, not universal vectors; the robot converts them to control signals before MuJoCo stepping.
5. The data-generation pipeline serializes history and scene/task metadata as HDF5. The saved artifact is a record, not proof that the task or replay succeeded.

### PnP / MimicGen bridge

`MolmoSpacesPnpEnv` presents a MolmoSpaces task to MimicGen. Its reset constructs a `JsonEvalTaskSampler` from a requested episode spec and can restore a source reset joint state. `MG_MolmoSpacesPickAndPlace` presents robot state, EEF pose, object state, and action conversion in the robot base frame.

`target_pose_to_action()` selects an execution representation including `tcp_delta`, `osc_pose`, and `joint_position`. Those paths are not interchangeable:

- `joint_position` is appropriate only after valid absolute IK and joint-limit/collision review.
- `tcp_delta` depends on local differential IK behavior.
- `osc_pose` uses Cartesian feedback and needs monitoring for singularities, saturation, and contact behavior.

The rollout bridge preserves gripper state across MimicGen subtask boundaries and inserts interpolated bridge waypoints. This is necessary for continuity but is a high-risk coupling point: the current logic assumes specific MimicGen execution ordering at the pickup-to-place boundary. Any change in subtask decomposition, MimicGen version, or waypoint timing requires a direct execution-order regression test.

`src/pnp/run_generation.py` owns source key selection, target-manifest range filtering, attempt inspection, de-duplication, subprocess logging, result aggregation, and JSONL provenance. It should remain the authoritative place for batch acceptance rather than allowing ad hoc rollout scripts to silently emit nominally equivalent data.

### Bimanual YAM is an independent workline

The YAM workflow is documented under `src/bimanual_yam/README.md` and `docs/worklines/`. It is distinct from Franka PnP and the older MolmoAct2 adapter.

- `browser_keyboard_teleop.py`: interactive browser keyboard controls, action generation for left arm, right arm, left gripper, and right gripper. This is the intended formal human source-demo entry point.
- `validate_tabletop_initialization.py`: checks object/table support and initialization provenance.
- `check_dual_object_reachability.py`: checks dual-object arm reachability and geometry diagnostics.
- `scripted_bimanual_source_demo.py`: produces a deterministic scripted/oracle diagnostic candidate. It is not a substitute for separately labelled human source demonstrations.

The scripted path records standard trajectory data plus `traj_0/mimicgen_yam` metadata: bimanual TCP/object state, low-level replay commands, layout, provenance, and strict-success observations.

## 5. Directory Guide

```text
molmospaces-integration/
  molmo_spaces/                         # Upstream framework code
    configs/                             # Experiment, robot, camera, task-sampler configs
    data_generation/                     # Config registry, CLI entry, rollout pipeline
    env/                                 # MuJoCo env, sensors, object/data views
    policy/                              # Policy interfaces and implementations
    renderer/                            # OpenGL/EGL renderer context setup
    tasks/                               # Task samplers and task success/reward logic
    utils/                               # MuJoCo scene/pose/asset utilities
  src/
    pnp/                                 # MimicGen-to-MolmoSpaces fixed-base PnP integration
    bimanual_yam/                        # Browser teleop, validation, scripted YAM source line
  docs/
    worklines/                           # Workline-specific operating and evidence documentation
  work/
    current/                             # Active, task-local work; inspect before acting
    completed/                           # Completed work records
    legacy/                              # Historical compatibility/debug material
  assets/, cache/, inputs/, runtime/     # Host/runtime artifacts; not source-of-truth code
```

### Where to find common concerns

| Need | Primary location |
|---|---|
| Add/inspect an experiment configuration | `molmo_spaces/configs/` and `data_generation/config_registry.py` |
| Trace a generic rollout | `data_generation/main.py`, `pipeline.py`, `tasks/task_sampler.py`, `tasks/task.py` |
| Change task sampling | task-specific sampler under `molmo_spaces/tasks/` |
| Change MuJoCo lifecycle or controls | `env/env.py`, robot config/controller classes, `tasks/task.py` |
| Inspect success predicate | concrete task, e.g. `tasks/pick_and_place_task.py` |
| Extend MimicGen PnP rollout | `src/pnp/generate_pick_place_rollout.py` |
| Change batch acceptance/provenance | `src/pnp/run_generation.py` |
| Change fixed target manifest | `src/pnp/sample_fixedbase_target_manifest.py` |
| Run YAM human source demo | `src/bimanual_yam/browser_keyboard_teleop.py` |
| Run YAM geometry diagnostics | `validate_tabletop_initialization.py`, `check_dual_object_reachability.py` |

## 6. Development Workflow

1. Recheck mutable facts first: `git rev-parse --short HEAD`, `git status --short`, environment path, assets, renderer, and live process state.
2. Keep a proposed change within the owning layer. Prefer `src/pnp/` or `src/bimanual_yam/` for integration behavior; modify `molmo_spaces/` only for a demonstrated reusable framework defect.
3. Record exact config name, source HDF5, target manifest, random seed, reset state, renderer environment, and output directory for every runtime experiment.
4. Use a smallest static gate before runtime: formatter/linter where configured, `python -m py_compile` for edited Python files, import/config-construction smoke where assets are not needed, then a bounded runtime gate.
5. Preserve unrelated dirty worktree changes. Never use reset/checkout/clean to make a review easier.
6. For behavioral claims, collect the deepest applicable evidence: valid initialization, real camera render, contact/action trace, task success metric, and fresh-reset replay where required.

The audit did not establish a CI pipeline, branch naming convention, or complete test suite. Do not invent one; inspect the current repository configuration before introducing CI, test commands, or style enforcement.

## 7. Evidence Contracts

### PnP target rollout acceptance

The generation layer is intended to gate an accepted sample on more than process completion: source/target provenance, target seed/layout identity, HDF5 integrity, action/layout hashing, video/artifact presence, success persistence, and direct/replay agreement. Review the current `generate_result.json`, result summary, and JSONL provenance schema before changing an acceptance rule.

### Bimanual YAM formal source-demo acceptance

The current formal target is stricter than a successful script invocation:

1. Capture must meet strict `2/2` success conditions.
2. HDF5 must include command-level, four-channel bimanual replay data and source provenance.
3. A fresh deterministic reset must reproduce the initial layout within `1e-6 m`.
4. Replayed low-level commands must reproduce final object positions within the configured tolerance (currently documented as `3 cm`).
5. `replay_gate_pass=true` is required, with actual rendered/video and HDF5 artifacts inspected separately.

This contract deliberately distinguishes a scripted expert diagnostic from a human source demonstration and artifact existence from demonstrated behavior.

## 8. Current Snapshot and Known Risks

### Existing dirty worktree boundary (audit snapshot)

The audited checkout had tracked changes in:

```text
molmo_spaces/configs/task_sampler_configs.py
molmo_spaces/env/object_manager.py
molmo_spaces/renderer/opengl_context.py
molmo_spaces/tasks/pick_and_place_task_sampler.py
molmo_spaces/tasks/pick_task_sampler.py
molmo_spaces/utils/mujoco_scene_utils.py
src/pnp/generate_pick_place_rollout.py
src/pnp/run_generation.py
src/pnp/runtime.py
src/pnp/sample_fixedbase_target_manifest.py
```

The audit observed `525` inserted and `72` deleted lines at that point. The YAM scripts themselves were not among those tracked changes. This list is only an audit-time boundary, not a substitute for `git status` before new development.

### High-risk interfaces

1. **MimicGen private sequencing assumption**: pickup-to-place bridge logic depends on execution-call ordering. Version or task segmentation changes can invalidate it silently.
2. **Coordinate-frame consistency**: EEF, object poses, source transforms, target transforms, IK targets, and reset state must remain in one documented frame. A plausible numerical result can be physically wrong after a frame mismatch.
3. **EGL / container rendering**: the repo includes a Mesa surfaceless EGL fallback (`EGL_MESA_platform_surfaceless`, `0x31DD`) because device-display initialization can fail in containers. Context creation is only a renderer smoke gate; verify actual official camera pixels.
4. **CLIP fallback**: `ObjectManager` was broadened from `NameError` to `Exception as exc` around CLIP use and logs `CLIP unavailable (%s), using dummy description scores.` This enables offline fixed-object initialization but can also mask unrelated CLIP failures. Treat it as an explicit fallback state, not normal semantic scoring.
5. **Fixed-orientation sampling**: `preserve_fixed_pickup_orientation: bool = False` was added to task-sampler configuration. Its interaction with fixed poses, grasp candidates, and reset determinism needs targeted testing.
6. **Artifact directories are not code**: untracked `.nltk_data/`, `.venv-bcrnn/`, `assets/`, `cache/`, `inputs/`, `install.exit`, and launcher/runtime remnants must not be swept into source commits without provenance and review.

## 9. First Contribution Checklist

- [ ] Read this document and the relevant workline README.
- [ ] Verify the live checkout commit and dirty diff; preserve unrelated changes.
- [ ] Identify whether the proposed change belongs upstream, PnP, or YAM.
- [ ] Trace one concrete config to sampler, task, robot action, and output HDF5.
- [ ] Identify the corresponding success predicate and evidence contract.
- [ ] Run a static/syntax gate for touched code before any expensive runtime job.
- [ ] For PnP changes, test source/target manifest identity, action conversion, and one bounded replay path.
- [ ] For YAM changes, run stationary gripper geometry/camera calibration before object contact; never infer open/close semantics from control limits alone.
- [ ] Store every bounded runtime attempt in a fresh named output directory with command, config, seed, result, and provenance.

## 10. Resources

- Upstream overview and setup: `README.md` and `pyproject.toml`.
- PnP workflow: `src/pnp/README.md`.
- Bimanual YAM public interface: `src/bimanual_yam/README.md`.
- Browser teleop workline: `docs/worklines/bimanual_yam_browser_teleop/README.md`.
- iTHOR YAM workline: `docs/worklines/ithor_bimanual_yam/README.md`.
- YAM scripted source baseline: `docs/worklines/bimanual_yam_source_baseline/README.md`.

## Audit Limits

This document is an architectural orientation and handoff record, not a runtime certification. It does not assert that the installed environment currently imports, renderer initialization currently works, source/target assets exist, a PnP rollout succeeds, or a YAM task satisfies its replay contract. Recheck those facts at execution time.
