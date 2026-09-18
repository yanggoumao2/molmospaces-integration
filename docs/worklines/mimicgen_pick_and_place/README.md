# MimicGen Pick-and-Place Workline

## Scope

This is the active Franka Droid / Pick-and-Place / in-process scripted-IK MimicGen workline. It uses replay-verified source trajectories, an independently sampled target-layout manifest, and real MolmoSpaces rollouts for acceptance. It is separate from RB-Y1 CuRobo planner datagen and from the archived broad 50-demo diagnostic.


The current cross-scene target control keeps the source provenance explicit: the
replay-verified source pool is from `house 1716`, while the independently reset target
is `house 3080` in the `procthor-objaverse` validation split. The recorded target uses
the `support_adapted_planar_pair` layout. One normal rollout passed final and
persistent post-hold success, but this remains a diagnostic control rather than a
formal cross-scene benchmark result or training-data release.

## Mainline contract

The primary route is one validated source demonstration expanded by MimicGen over
multiple target layouts. The target manifest supplies the layout variation; it is not
replaced by collecting a new source for every target. For the generic runner, use
`--mode whole-source --source-count 1`. For FloorPlan1 bimanual YAM,
`src/pnp_bimanual_yam/run_datagen.py` converts one enriched source into right/left
MimicGen inputs and keeps both arms inside one physical reset.

## Active Entry Points

The active code is intentionally limited to [`src/pnp/`](../../../src/pnp/):

- `run_source_hdf5_pipeline.py`: parameterized source candidate selection, replay, conversion, and validation.
- `run_generation.py`: parameterized MimicGen generation runner; the one-source mainline uses `--mode whole-source --source-count 1`, while multi-source comparisons may use `--mode per-subtask`.
- `generate_pick_place_rollout.py`: one simulator rollout primitive.
- `select_source_pool.py`, `replay_source_candidate.py`, `convert_source_hdf5.py`, and `validate_robomimic_source_hdf5.py`: the source-HDF5 pipeline stages.
- `sample_fixedbase_target_manifest.py` and `validate_fixedbase_target_manifest.py`: target-layout stages.

The mainline uses one validated source demonstration. The 17-demo replay-verified pilot is an optional multi-source comparison; its count is an input to `run_generation.py`, not a module or directory name.

## Minimal Commands

```bash
python src/pnp/run_source_hdf5_pipeline.py --help
python src/pnp/sample_fixedbase_target_manifest.py --help
python src/pnp/validate_fixedbase_target_manifest.py --help
python src/pnp/run_generation.py --help
python src/pnp/generate_pick_place_rollout.py --help
```

Use an explicit source HDF5 and target manifest for every run. New experiments must not add fixed-count shell launchers or embed Python in shell scripts.

## Evidence Boundary

Source HDF5 validation and a simulator process exit are prerequisite checks, not task-success evidence. Accepted generated demonstrations require a real simulator rollout, final success, persistent post-hold success, non-empty videos, matching target layout, and no duplicate action or layout fingerprint.

`runtime/`, `results/`, and `artifacts/` retain prior evidence at their original paths. Historical scripts and snapshots live under [`archive/pnp/`](../../../archive/pnp/) and are not active entrypoints.
