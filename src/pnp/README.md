# Pick-and-Place Augmentation Pipeline

This directory is the supported Python implementation of the reusable
MolmoSpaces × MimicGen Pick-and-Place data-augmentation workflow. Public
experiments are configuration-driven and scenario parameters must be supplied
explicitly; active source files do not encode a private machine path, a fixed
scene, or a particular object pair.

**Start here:** [`docs/pnp-data-augmentation.md`](../../docs/pnp-data-augmentation.md).

## Architecture

```text
one validated source HDF5 + validated target manifest
                         |
          run_experiment.py + JSON configuration
                         |
                 run_generation.py
                         |
         generate_pick_place_rollout.py
                         |
 per-attempt artifacts + JSONL provenance + HDF5 + summary
```

## Mainline: one source to many target layouts

This companion Franka adapter supports one validated source demonstration plus an immutable target
manifest. The completed bimanual mainline is documented in the [YAM guide](../pnp_bimanual_yam/README.md). MimicGen applies the source trajectory to every target entry, so target
layout variation provides expansion without collecting a new source for each layout.
Use the direct runner with an explicit one-demo source pool:

```bash
"$MOLMOSPACES_PYTHON" src/pnp/run_generation.py \
  --work runtime/one_source_expansion \
  --source-hdf5 /path/to/validated_source_one_demo.hdf5 \
  --target-manifest /path/to/target_manifest.json \
  --mode whole-source \
  --source-count 1 \
  --diagnostic \
  --target-success 10 \
  --max-attempts 30 \
  --run-label one_source_mimicgen
```

`--source-count 1` checks the HDF5 demo count; it does not truncate a larger pool.
The whole-source Franka control retains its diagnostic classification. `per-subtask`
can select from either one source or a larger pool. The FloorPlan1
bimanual YAM implementation follows the same contract in
`src/pnp_bimanual_yam/run_datagen.py`: one enriched source is converted to one
source per arm, then the right and left arms execute sequentially in one physical
reset while each target-layout result is checkpointed.

| Module | Responsibility |
| --- | --- |
| `select_source_pool.py` | Select source candidates from supported source data. |
| `replay_source_candidate.py` | Replay a source candidate and record MimicGen datagen fields. |
| `convert_source_hdf5.py` | Convert replay-accepted sources into a MimicGen/robomimic HDF5. |
| `validate_robomimic_source_hdf5.py` | Validate source schema, finite values, alignment, and provenance. |
| `sample_fixedbase_target_manifest.py` | Sample reset-only target `EpisodeSpec` records from explicit scenario arguments. |
| `validate_fixedbase_target_manifest.py` | Verify target-manifest schema, uniqueness, and scenario consistency. |
| `run_source_hdf5_pipeline.py` | Orchestrate source selection, replay, conversion, and validation. |
| `run_experiment.py` | Validate a portable JSON generation configuration and invoke the batch runner. |
| `run_generation.py` | Batch orchestration, input hashes, deduplication, JSONL records, and summary. |
| `generate_pick_place_rollout.py` | Single real simulator rollout primitive. |

## Boundaries

- Python contains data, simulator, validation, and orchestration logic.
- `scripts/pnp/` is the shell boundary: it only locates the repository and
  interpreter, then invokes Python. It contains no Python heredoc or dataset
  policy.
- `configs/pnp/` stores portable example configurations. Copy an example for a
  run; never encode a local path or scenario in a tracked launcher.
- `archive/pnp/` contains historical one-off experiments and debug snapshots.
  They are preserved for provenance and are not supported APIs.

## Public entrypoints

```bash
# Inspect all CLI contracts first.
"$MOLMOSPACES_PYTHON" src/pnp/run_source_hdf5_pipeline.py --help
"$MOLMOSPACES_PYTHON" src/pnp/sample_fixedbase_target_manifest.py --help
"$MOLMOSPACES_PYTHON" src/pnp/validate_fixedbase_target_manifest.py --help
"$MOLMOSPACES_PYTHON" src/pnp/run_experiment.py --help

# Validate a copied configuration without starting a rollout.
scripts/pnp/run_generation.sh configs/pnp/generation.example.json --dry-run
```

The example intentionally references placeholder input artifacts. It is a
configuration/schema smoke test, not a runnable dataset collection command.
Use a validated source HDF5 and target manifest to run a real smoke.


## Current target-layout control

A recorded cross-scene control reuses the `house 1716` replay-verified source pool
on an independently reset `house 3080` target in the `procthor-objaverse` validation
split. Its target manifest uses the `support_adapted_planar_pair` layout and records
explicit pickup/receptacle identities and reset poses. One normal rollout reached
final success and persistent post-hold success. Treat this as a diagnostic control
for the target-layout route only; it is not a formal cross-scene success-rate result,
full 6-D rigid migration claim, or training-ready dataset.

## Evidence contract

Successful process completion is insufficient. `run_generation.py` records an
attempt only as accepted when task behavior, requested target, selection mode,
post-hold persistence, nonempty media, HDF5 persistence, and action/layout
uniqueness all pass. Dataset-level eligibility additionally requires the target
acceptance count and generated aggregate HDF5. See the full artifact contract
and reproducibility requirements in the workflow guide.
