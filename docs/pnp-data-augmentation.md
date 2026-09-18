# Configuration-Driven Pick-and-Place Data Augmentation

This guide is the supported entrypoint for building a reproducible
MolmoSpaces × MimicGen Pick-and-Place dataset. It deliberately separates
scenario definition, Python data logic, and shell environment setup.

## What this repository provides

- **Python CLIs** in `src/pnp/` for source preparation, reset-manifest
  sampling, rollout generation, validation, and artifact accounting.
- **JSON configurations** in `configs/pnp/` that record a run's portable
  inputs and parameters without embedding them in a script name.
- **Thin shell launchers** in `scripts/pnp/`. They resolve an interpreter and
  invoke Python; they contain no dataset logic, Python heredocs, or acceptance
  policy.
- **Archived historical utilities** in `archive/pnp/`, retained for audit but
  excluded from the supported public API.

## Prerequisites

1. Clone recursively enough to obtain this repository's tracked source:
   ```bash
   git clone <repository-url>
   cd molmospaces-integration
   ```
2. Create a Python environment compatible with the repository's pinned
   MolmoSpaces dependencies. Install the repository according to the root
   README.
3. Fetch the separately licensed MimicGen and robomimic source trees:
   ```bash
   bash tools/setup_mimicgen_dependency.sh
   ```
   `vendor/` is intentionally ignored by Git. Check the exact upstream commit
   recorded by the setup script before changing it.
4. Export only portable paths; do not edit a tracked script for local paths:
   ```bash
   export MOLMOSPACES_PYTHON=/path/to/your/python
   export MIMICGEN_ROOT="$PWD/vendor/mimicgen"
   export ROBOMIMIC_ROOT="$PWD/vendor/robomimic"
   export PYTHONPATH="$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT:${PYTHONPATH:-}"
   export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
   export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
   export MOLMOSPACES_NLTK_DATA="$NLTK_DATA"
   ```

GPU rendering requires a Linux host with a correctly installed NVIDIA EGL
stack. CPU rendering can be useful for import/configuration diagnosis, but is
not a performance-equivalent data-collection configuration.

## Supported workflow

### Mainline contract: one source, many targets

The supported mainline starts with one validated source demonstration. MimicGen receives that complete one-demo source and transforms it for each immutable target-layout entry; the target manifest, not a second source collection, provides diversity. The same contract is used by the generic Franka runner and by the sequential bimanual YAM runner.

For the generic runner:

```bash
"$MOLMOSPACES_PYTHON" src/pnp/run_generation.py \
  --work runtime/one_source_expansion \
  --source-hdf5 /path/to/validated_source_one_demo.hdf5 \
  --target-manifest /path/to/target_manifest.json \
  --mode whole-source \
  --source-count 1 \
  --target-success 10 \
  --max-attempts 30 \
  --run-label one_source_mimicgen
```

`whole-source` is intentional here: with `--source-count 1`, MimicGen sees exactly one validated source demo for every target. Use `per-subtask` only when a multi-source comparison is the actual experiment.

For FloorPlan1 bimanual YAM, one enriched source is converted to one right-arm and one left-arm MimicGen source, then both arms run sequentially inside one physical reset:

```bash
PYTHONPATH=.:vendor/mimicgen:vendor/robomimic "$MOLMOSPACES_PYTHON" src/pnp_bimanual_yam/run_datagen.py \
  --src /path/to/source_demo_enriched.h5 \
  --num-attempts 16 \
  --sampling-design lhs \
  --xy-jitter 0.10 \
  --manifest runtime/one_source_yam/manifest.json \
  --success-dir runtime/one_source_yam/successes
```

This expands one source across target layouts; it does not create a new source demonstration per target.

### 1. Prepare a validated source dataset

Run the source pipeline with explicit input/output paths. It invokes selection,
deterministic replay, conversion, and source-HDF5 validation as separate Python
operations:

```bash
"$MOLMOSPACES_PYTHON" src/pnp/run_source_hdf5_pipeline.py --help
```

Before reuse, verify the source HDF5 has the expected demo count, finite arrays,
MimicGen datagen fields, action alignment, and recorded provenance.

### 2. Sample independent target resets

The target sampler intentionally has no hidden scene/object/base-pose defaults.
Supply every scenario-defining value explicitly and validate its output before
rollouts:

```bash
"$MOLMOSPACES_PYTHON" src/pnp/sample_fixedbase_target_manifest.py --help
"$MOLMOSPACES_PYTHON" src/pnp/validate_fixedbase_target_manifest.py --help
```

A target manifest is an input artifact: keep it immutable after a generation
run begins. It stores the frozen reset state and a layout SHA-256 per target.


For the current cross-scene control, the source pool remains from `house 1716` while
the target manifest describes an independently reset `house 3080` layout in the
`procthor-objaverse` validation split. This separation is part of the provenance
record: do not describe the target as an additional source demonstration. The
recorded `support_adapted_planar_pair` control reached final and persistent success
for one rollout, which is useful path-validation evidence but not a dataset-level
success-rate claim.

### 3. Create and validate a generation configuration

Copy the example rather than editing code:

```bash
cp configs/pnp/generation.example.json configs/pnp/my_run.json
```

Set `repository_root`, `work_dir`, the validated source HDF5, the validated
target manifest, and the desired generation controls. Paths relative to the
configuration file are supported; source/manifest paths may also be absolute.

Validate before allocating a GPU or writing data:

```bash
scripts/pnp/run_generation.sh configs/pnp/my_run.json --dry-run
```

Run a one-target smoke first. Only after reviewing its artifacts should you
raise `target_success`, target range, or maximum attempts:

```bash
scripts/pnp/run_generation.sh configs/pnp/my_run.json
```

`whole-source` with `--source-count 1` is the documented one-source mainline. The
configuration wrapper still requires `generation.diagnostic=true` whenever it is used
for a whole-source compatibility run; the direct CLI above makes the validated
one-source contract explicit. `per-subtask` remains available for multi-source
recombination studies.

## Artifact contract and acceptance

Each run receives a new log directory containing `collector.log`, per-attempt
logs, `attempts.jsonl`, `accepted.jsonl`, action/layout hash registries, and
`summary.json`. Every record includes source and target-manifest SHA-256 values.

An accepted rollout must independently satisfy all of the following:

1. the child rollout completes without error;
2. selection mode and requested target are verified;
3. final task success persists through the configured post-hold window;
4. required video artifacts are nonempty;
5. replay and aggregated HDF5 persistence are nonempty;
6. action and target-layout hashes are unique among accepted outputs.

A command exit code, a video, or a nonempty HDF5 alone is not a valid dataset
acceptance claim. Review HDF5 schema/count, simulator behavior, and provenance
separately.

## Reproducibility record

For every reported dataset, retain the configuration JSON unchanged alongside
its output directory, plus repository commit, `git diff`, interpreter/package
versions, input SHAs, GPU/renderer information, command line, and validation
reports. Do not publish a result as reproducible if it depends on unrecorded
private paths, cached resources, manually patched vendor code, or a local
uncommitted change.

## Common failures

- **Import or CLIP/cache error:** verify `PYTHONPATH` points to the intended
  MimicGen/robomimic checkout and that offline caches are available. Different
  vendor trees are not automatically interchangeable.
- **Renderer failure:** verify NVIDIA EGL on native Linux; do not treat an
  import-only CPU run as equivalent to GPU data collection.
- **Zero accepted rollouts:** inspect target-manifest validity, per-stage task
  traces, source compatibility, and generated HDF5 persistence before retrying
  or changing acceptance thresholds.
- **Resume confusion:** reuse the same work directory only when its input SHAs
  and configuration are identical. Otherwise choose a new `work_dir` and label.
