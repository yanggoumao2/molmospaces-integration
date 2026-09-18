# One-source bimanual YAM MimicGen pipeline

This is the repository's mainline: **one validated FloorPlan1 source demonstration → MimicGen → multiple target-layout rollouts**. The pipeline is implemented and has completed a 16-layout simulation pilot with 6 strict successes.

## Verified pilot

The [public evidence summary](../../results/workline_index/bimanual_yam_one_source_pilot.json) records the 2026-09-15 pilot. The original manifest and six nonempty success archives were rechecked on 2026-09-18.

| Setting / result | Value |
|---|---|
| Source episodes | 1; right and left HDF5 inputs each contain `demo_0` from this episode |
| Target design | 16-point LHS over potato, tomato, and plate XY offsets |
| Perturbation range | ±0.10 m for each XY coordinate |
| Seed / interpolation | `20260915` / 40 steps |
| Completed / strict successes | 16 / 6 (37.5%) |
| Stored outputs | Per-attempt manifest; 6 compressed NPZ trajectory archives |

This is observed success in a bounded simulation pilot. HDF5 replay/export, dataset provenance/deduplication, continuous-motion audits, and larger-scale validation remain separate acceptance gates.

## Pipeline

1. Start from the corrected `source_demo_enriched.h5`, including both arms' grasp, carry, release, and stow phases.
2. `convert_source_hdf5.py` creates `source_right.hdf5` and `source_left.hdf5`, each with one `demo_0` and MimicGen datagen fields. EEF poses come from forward kinematics of the replay commands, aligned with the gripper stream.
3. `run_datagen.py` samples target layouts. MimicGen transforms source waypoints relative to the food and plate; the adapter uses source-seeded IK and a per-layout carry blend.
4. The right arm places the potato. The left arm then places the tomato in the same physical episode: the right generator owns the single reset and the left generator's reset is suppressed.
5. After settling, both foods must satisfy `released`, `plate_supported`, and `not_dropped`. Every attempt checkpoints its layout and result; only strict successes produce trajectory NPZ files when `--success-dir` is set.

The two arm inputs are views of one source episode. The runner uses `select_src_per_subtask=False` with `demo_keys=["demo_0"]` for each arm.

## Required runtime inputs

- Activate the project Python 3.11 environment and install the [repository dependencies](../../README.md#franka-datagen-quick-start). Fetch the pinned MimicGen/robomimic trees with `bash tools/setup_mimicgen_dependency.sh` and install them in that environment.
- Supply the FloorPlan1 scene bundle used by the source. `YAM_FOOD_TASK_DIR` points to the directory containing `correction_assembly/assemble_combined_scene.py`; its referenced scene XML, robot, and object assets must also be available. `_scene.py` supplies the repository-relative default used by the existing runtime workspace.
- Supply the corrected enriched source via `--src` or `YAM_SOURCE_HDF5`. The current `task_spec.py` boundaries and carry-blend windows are aligned to that source: right `[815, 5777)`, left `[5777, 11311)`. A different source needs matching segmentation and validation.

The scene bundle, source HDF5, and generated data are runtime inputs excluded from Git. A clean clone needs these inputs before a real rollout; this guide documents the working simulation pipeline and its dependency boundary.

## Run a bounded expansion

Run from the repository root after preparing those inputs. The source path below is a placeholder. Choose a fresh working directory for each source/configuration.

```bash
PYTHONPATH="$PWD:$PWD/vendor/mimicgen:$PWD/vendor/robomimic:${PYTHONPATH:-}" \
python src/pnp_bimanual_yam/run_datagen.py \
  --src /path/to/validated/source_demo_enriched.h5 \
  --workdir runtime/one_source_yam \
  --num-attempts 16 \
  --seed 20260915 \
  --sampling-design lhs \
  --xy-jitter 0.10 \
  --num-interpolation-steps 40 \
  --manifest runtime/one_source_yam/manifest.json \
  --success-dir runtime/one_source_yam/successes
```

`--workdir` owns the cached per-arm source files. The runner reuses them if they already exist, so changing `--src` in an old working directory does not reconvert them. Use a fresh directory for a changed source; the earlier left-arm cache had 5014 samples, while the corrected left-arm window has 5534. Keep `--num-interpolation-steps 40` explicit because the runner's CLI default is 5.

To resume an interrupted run, repeat exactly the same command with `--resume`, retaining the source, cached arm files, seed, layout design, and all control settings. The manifest checks only its recorded configuration fields; retain the complete launch command alongside it, including the interpolation setting.

## Outputs and review

| Output | Meaning |
|---|---|
| `source_right.hdf5`, `source_left.hdf5` | One converted source per arm; not generated target trajectories |
| `manifest.json` | Requested and settled target layouts, per-arm status, strict success reports, and aggregate counts |
| `successes/*.npz` | Successful per-arm actions and simulator states (`qpos`, `qvel`, `ctrl`, `time`) |

Use the manifest's `n_completed`, `n_success`, and per-attempt reports to assess the run. A zero exit code alone does not mean a successful expansion. This runner does not automatically export a training HDF5 or render videos; those are separate validation/export steps. The generic Franka runner has a different [artifact contract](../../docs/pnp-data-augmentation.md#artifact-contract-and-acceptance).
