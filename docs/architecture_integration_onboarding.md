# MolmoSpaces Integration Architecture Onboarding

> **Scope.** This is a maintainer-oriented architecture map for this integration checkout. It records a read-only audit completed on **2026-09-08 (Asia/Shanghai)** against commit `c64d3ba`. It is not a claim about current runtime health, dataset validity, benchmark performance, or the state of later uncommitted changes. Re-check mutable state before executing work.

## 1. Project identity and boundaries

- **Primary stack:** Python 3.11+ robotics/simulation project (`molmo-spaces` 0.2.0) using MuJoCo, Pydantic configuration, h5py, PyTorch and optional CuRobo/MimicGen dependencies.
- **Project type:** upstream MolmoSpaces framework plus a local integration layer for MimicGen Pick-and-Place (PnP) and a distinct bimanual YAM source-demo/teleoperation workline.
- **Repository root:** the integration repository checkout. The canonical long-lived 4090 checkout is elsewhere; do not assume the two trees or worktrees are identical.
- **Rule of interpretation:** `molmo_spaces/` is primarily framework code. `src/pnp/` and `src/bimanual_yam/` are integration/workline code. A behavior visible in one layer is not automatically an upstream framework guarantee.
- **Large artifacts:** raw HDF5, videos, run directories, caches, backup scripts, PID files and launch logs are deliberately outside normal source control. Their presence is evidence of local activity, not a reproducible result.

## 2. Architecture at a glance

```text
                 Python module entry/config name
                              |
                              v
       molmo_spaces.data_generation.main / config_registry
                              |
                              v
                     ParallelRolloutRunner
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
  BaseMujocoTaskSampler                policy / controller setup
  - choose house / episode                     |
  - load or reuse MuJoCo scene                 |
  - sample objects, task config                v
             |                         robot.update_control()
             v                         robot.compute_control()
       CPUMujocoEnv / model,data                |
             |                                  v
             v                           MuJoCo physics steps
        BaseMujocoTask <------------------------+
  reset -> sensors/policy -> initial observation
  step(action) -> observations/rewards/done/history
             |
             v
       HDF5 trajectory + metadata + optional video

Integration worklines:

  Source HDF5 + target manifest                    Browser teleop / scripted expert
                 |                                             |
                 v                                             v
  src/pnp/generate_pick_place_rollout.py       src/bimanual_yam/*.py
                 |                                             |
                 v                                             v
  MG_MolmoSpacesPickAndPlace                standard task history + YAM audit group
                 |                                             |
                 +----------- MolmoSpaces task/env ------------+
```

### Dependency direction

```text
configs -> task sampler -> scene/object manager -> MuJoCo env -> task -> sensors/history
configs -> policy/robot -> task.step -> MuJoCo env
src/pnp -> MimicGen APIs + MolmoSpaces task/env/robot interfaces
src/bimanual_yam -> MolmoSpaces task/env/robot interfaces
run_generation -> source HDF5 + target manifest + rollout result artifacts
```

Foundation modules are configuration, scene/object management, MuJoCo environment and task abstractions. Task samplers orchestrate scene lifecycle. `src/pnp/` and `src/bimanual_yam/` are leaves that impose workflow-specific data and acceptance contracts on the framework.

## 3. Directory guide

```text
molmo_spaces/                 # Upstream framework implementation
  configs/                    # Pydantic experiment, robot, camera, task-sampler configs
  data_generation/            # Config registry, CLI module entry, pipeline/runner
  env/                        # BaseMujocoEnv, CPUMujocoEnv, sensors, object/data views
  tasks/                      # Task samplers and task success/reward/history behavior
  policy/                     # Policy interfaces and planner implementations
  renderer/                   # EGL/OpenGL context and camera rendering
  robots/                     # Robot/controller integration
src/
  pnp/                        # MimicGen Pick-and-Place adaptation, manifests and orchestration
  bimanual_yam/               # Browser teleop, initialization/reachability, scripted source demos
  molmoact2_legacy/           # Historical compatibility path; not the current PnP/YAM route
docs/                         # Public framework and workline documentation
  worklines/                  # Workline boundaries and setup guides
mlspaces_tests/               # Component, data-generation, scene and CuRobo test families
tools/                        # Dependency/setup helpers, including MimicGen pinning
results/                      # Lightweight tracked manifests/summaries; not large artifacts
runtime/, runs/, cache/, ...  # Local runtime state when present; inspect provenance before use
```

## 4. Primary entry points and boot sequence

### A. Upstream data generation

**Entry:** `python -m molmo_spaces.data_generation.main <RegisteredConfigName>`.

1. `config_registry.py` resolves a named registered experiment configuration.
2. The configuration combines task, robot, sensor/camera, policy, output and rollout settings.
3. `ParallelRolloutRunner` schedules house/episode work and owns parallel rollout boundaries.
4. `BaseMujocoTaskSampler.sample_task()` advances/selects a house, loads or reuses its scene, initializes task configuration and returns a task instance.
5. The task runs policy actions through the robot controller and `CPUMujocoEnv`; trajectory history is exported to the data writer.

Important framework files:

| File | Responsibility | Change impact |
|---|---|---|
| `molmo_spaces/data_generation/main.py` | CLI/module entry and output-path semantics | Broad: affects official generation entry |
| `molmo_spaces/data_generation/pipeline.py` | Rollout orchestration and serialization boundaries | Broad: affects artifact lifecycle |
| `molmo_spaces/configs/task_sampler_configs.py` | Task-sampler schema and registered behavior knobs | Broad: configuration compatibility |
| `molmo_spaces/tasks/task_sampler.py` | House advancement, scene load/reuse, task creation | Very broad: state/provenance boundary |
| `molmo_spaces/env/env.py` | MuJoCo model/data lifecycle and physics stepping | Very broad: simulator semantics |
| `molmo_spaces/tasks/task.py` | reset/step/history/terminal semantics | Very broad: all policy data |
| `molmo_spaces/tasks/pick_and_place_task.py` | Pick-and-place reward/success metrics | Narrower task semantics |

### B. MimicGen PnP generation

**Key entry:** `src/pnp/generate_pick_place_rollout.py`.

- `src/pnp/runtime.py` supplies process/environment helpers and result inspection support.
- `src/pnp/run_generation.py` is the batch/provenance coordinator: validates the source pool, reads target-manifest seed coverage, checks existing attempts, deduplicates results and writes JSONL summaries.
- `src/pnp/sample_fixedbase_target_manifest.py` creates/validates fixed-base target seed selections.
- `MG_MolmoSpacesPickAndPlace` adapts MimicGen calls to MolmoSpaces reset, observation, robot-frame pose and low-level action interfaces.

The PnP workflow must treat source demonstrations, target manifest, resulting HDF5, video and result JSON as one provenance unit. Do not substitute a nonempty output directory for accepted-generation evidence.

### C. Bimanual YAM source demos and diagnostics

**Key entries:**

- `src/bimanual_yam/browser_keyboard_teleop.py`: human-operated browser keyboard teleoperation.
- `src/bimanual_yam/validate_tabletop_initialization.py`: scene/object/table support initialization diagnostics.
- `src/bimanual_yam/check_dual_object_reachability.py`: two-object reachability diagnostics.
- `src/bimanual_yam/scripted_bimanual_source_demo.py`: scripted/oracle source candidate and replay audit path.
- `docs/worklines/bimanual_yam_browser_teleop/README.md` and `docs/worklines/ithor_bimanual_yam/README.md`: workline-specific operating boundary.

This is not a two-arm extension of the MimicGen PnP implementation. Browser teleop is for formally labelled source collection; scripted expert execution is a diagnostic/bootstrap path unless its provenance and acceptance conditions explicitly say otherwise.

## 5. Key data and control flows

### 5.1 Task sampler -> task -> environment

`BaseMujocoTaskSampler.sample_task()` freezes the initial task configuration when needed, advances or pins the requested house, and decides whether the MuJoCo scene must be loaded or can be reused. Scene reuse restores the scene-level preset task configuration instead of resampling arbitrary state.

`BaseMujocoTask.reset()` clears cumulative/trajectory state, resets stateful sensors and a registered policy, samples the initial observation, and freezes the task config with the initial observation. `BaseMujocoTask.step(action)` normalizes single/batch action input, routes it through `robot.update_control()`, repeatedly calls `robot.compute_control()` and environment physics stepping, then polls sensors and caches action/observation/reward/terminal information. `get_history()` is the canonical in-memory trajectory record.

`PickAndPlaceTask` adds pickup/receptacle object selection, relevant sensors and success information. Its success predicate involves placement/support semantics; proximity alone is insufficient.

### 5.2 PnP source -> target rollout -> accepted result

```text
source HDF5 data group -> sorted source_keys
                         -> target manifest seeds / valid target range
                         -> MimicGen transformed subtask trajectories
                         -> MG_MolmoSpacesPickAndPlace reset/action bridge
                         -> rollout HDF5, video, result JSON
                         -> artifact inspection + uniqueness/provenance summary
```

The adapter exposes robot-base-frame end-effector/object poses and turns target poses into one of several action families (`joint_position`, `tcp_delta`, or `osc_pose`). The customized waypoint executor maintains gripper state across MimicGen subtask boundaries, interpolates bridge waypoints and records context for later checks.

### 5.3 YAM capture -> HDF5 -> fresh-reset replay

The scripted path writes standard trajectory data and, for `traj_0`, a `mimicgen_yam` audit group containing YAM-specific command/context metadata. The formal contract is stronger than a script exit code:

1. strict capture success for both required object operations (`2/2`);
2. complete low-level command/HDF5 record and metadata;
3. deterministic fresh reset of the intended scene/layout;
4. replay of the recorded commands;
5. matching initial layout and sufficiently close final object state; and
6. an explicit `replay_gate_pass=true` record.

A render, an IK solution, an HDF5 file, or a successful process alone is diagnostic evidence, not a valid source demo.

## 6. Local conventions and development workflow

- Python uses `snake_case` modules/functions/variables and `PascalCase` classes. Pydantic-style typed config models are the public schema boundary.
- Ruff is configured in `pyproject.toml` for Python 3.11 with line length 100. It enables E/F/UP/B/SIM/I/W/ANN families with documented exemptions.
- Tests live under `mlspaces_tests/`, grouped into component, data-generation, CuRobo and scene families. Relevant existing examples include `test_json_eval_task_sampler.py`, `test_franka_pick_and_place.py`, `test_kinematics.py`, and `test_rby1_pnp.py`.
- The optional `mujoco` dependency group supplies MuJoCo. PnP setup also requires `tools/setup_mimicgen_dependency.sh`, which pins the external MimicGen/robomimic checkout under `vendor/`.
- Treat `src/pnp/` and `src/bimanual_yam/` as public integration code; use the existing `docs/worklines/` separation rather than making PnP, YAM and legacy MolmoAct2 appear to be one runtime.

### Narrow verification sequence for a change

1. Read `git status --short`, `git diff --stat`, and targeted diffs; do not overwrite existing local changes.
2. Determine whether the change belongs in upstream framework code or an integration workline.
3. Run syntax/static checks first, for example `python -m py_compile <changed Python files>` and targeted `ruff check` where the environment has Ruff installed.
4. Run the smallest relevant existing test or a purpose-built non-generative smoke test.
5. For rendering changes, require a real nonblank official camera artifact in the target runtime; EGL initialization alone is insufficient.
6. For PnP/YAM behavior, use the workline acceptance contract, including reset provenance, contacts/support, replay and result files as applicable.

The full CI/deployment pipeline was not audited in this pass. Do not claim a CI gate exists without reading the current repository configuration.

## 7. Known integration risks and current worktree boundary

### High-risk interfaces

1. **PnP subtask timing.** The customized PnP executor currently relies on MimicGen execution-call ordering around pickup/place (`execute_call_index == 4/5`). A different MimicGen decomposition can silently change the semantic boundary. Add/maintain an explicit test before changing subtask construction.
2. **Frame and action semantics.** Object-relative transformations require all pose inputs in the robot base frame. Keep `joint_position`, TCP-delta and OSC behavior distinct; a nominally valid IK output does not prove a safe trajectory.
3. **YAM gripper semantics.** Do not infer open/close from ctrl range/sign. The exact assembled robot needs a stationary geometry plus official-camera `open -> close -> open` calibration before any grasp claim.
4. **Container rendering.** The checkout includes a Mesa surfaceless EGL fallback path. This is compatibility code, not rendering validation; test a real camera output in the actual runtime.
5. **Offline object descriptions.** The modified `ObjectManager` fallback catches broad exceptions around CLIP availability. It improves offline fixed-object operation but may hide unrelated faults; preserve a distinguishable failure signal before treating it as production behavior.
6. **Evidence inflation.** A process exit, action trace, image, HDF5 or JSON is not a task-success metric. Source-demo and generation claims need their stated contacts, strict success, video/provenance and fresh-reset replay gates.

### Audited worktree snapshot

At the audit snapshot, commit `c64d3ba` had ten tracked modified files and `525` additions / `72` deletions:

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

The checkout also contained untracked environments, assets/caches, inputs, runtime directories, launch scripts/logs, a partial checkout and dated backup variants of PnP scripts. Those are not source changes to reset, stage or treat as a clean reproducibility baseline. This onboarding document itself is a subsequent documentation addition and changes the worktree status.

## 8. Where to start for common changes

| Goal | Start here | Required next check |
|---|---|---|
| Add/alter an official task or sampler option | `molmo_spaces/configs/`, relevant `tasks/*_task_sampler.py` | Config registration, sampler reuse and task reset behavior |
| Change MuJoCo/task behavior | `env/env.py`, `tasks/task.py`, concrete task file | Action cache/history, sensor timing and success predicate |
| Change PnP rollout behavior | `src/pnp/generate_pick_place_rollout.py` | MimicGen subtask semantics, frame/action mode, source/target provenance |
| Change batch generation/acceptance | `src/pnp/run_generation.py`, `runtime.py` | Result inspection, deduplication and JSONL summary fields |
| Change fixed target selection | `src/pnp/sample_fixedbase_target_manifest.py` | Manifest schema and seed-range compatibility |
| Change human source collection | `browser_keyboard_teleop.py` plus browser workline README | Teleop key semantics, HDF5 schema and formal source provenance |
| Change YAM scripted diagnostic | `scripted_bimanual_source_demo.py` | Strict 2/2 capture and fresh-reset replay, not only script completion |
| Diagnose scene/table layout | `validate_tabletop_initialization.py`, `check_dual_object_reachability.py` | Named support/contact, collision and reachability evidence |

## 9. First-session checklist

- [ ] Read this guide, `docs/worklines/README.md`, and the target workline README.
- [ ] Inspect live `git status --short`, current commit and relevant diff before making any change.
- [ ] Identify whether the task belongs to upstream framework, PnP, YAM or legacy MolmoAct2.
- [ ] Confirm environment/resource paths and optional dependencies without exposing credentials.
- [ ] Trace one path end-to-end: config -> sampler -> task -> environment -> history/output.
- [ ] For PnP, inspect source HDF5, target manifest and result acceptance fields together.
- [ ] For YAM, verify the correct task-local robot base, official camera and gripper calibration before motion.
- [ ] Run the smallest relevant static/test/render/behavior gate, then state precisely what it proves and does not prove.

## 10. Related documentation

- `README.md`: installation and top-level framework notes.
- `docs/code_structure.md`: broader upstream package organization.
- `docs/data_format.md` and `docs/data_processing.md`: trajectory/data handling.
- `docs/worklines/README.md`: active, legacy and archived workline boundaries.
- `docs/worklines/mimicgen_pick_and_place/README.md`: PnP-specific setup/usage.
- `docs/worklines/bimanual_yam_browser_teleop/README.md`: browser teleop workflow.
- `docs/worklines/ithor_bimanual_yam/README.md`: iTHOR YAM evidence and execution boundary.
- `mlspaces_tests/README.md`: test suite information.
