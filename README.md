<div align="center">
  <img src="media/banner.png" alt="MolmoSpaces x MimicGen: Franka pick-and-place trajectory augmentation" width="980" />
</div>

<p align="center">
  <a href="README.md"><strong>English</strong></a> &nbsp;|&nbsp; <a href="README_zh.md">中文</a>
</p>

<h1 align="center">MolmoSpaces x MimicGen</h1>

<p align="center">
  <strong>Controlled pick-and-place trajectory augmentation in simulation.</strong><br />
  From validated Franka source demonstrations to transformed target layouts, with explicit replay, persistence, and uniqueness gates.
</p>

<p align="center">
  <code>Franka + DROID cameras</code> &nbsp; <code>whole-source / per-subtask transfer</code> &nbsp; <code>strict artifact validation</code>
</p>

<p align="center">
  <img src="media/gif/heterogeneous_generated_examples.gif" alt="Generated Franka pick-and-place rollouts" width="720" />
</p>

<p align="center">
  <a href="media/heterogeneous_generated_examples.mp4">Open full-size generated rollouts</a>
</p>

## The primary workline

```text
One validated source demonstration  ->  MimicGen expansion across target layouts  ->  Replay, video, HDF5, and uniqueness checks
```

The repository focuses on a reproducible Franka Pick-and-Place route. It separates source collection, target-layout sampling, and generated rollout acceptance so that a runnable process, a saved artifact, and a strict task outcome are never treated as interchangeable evidence.

<table>
  <tr>
    <td width="50%" align="center"><img src="media/gif/source_candidate_example.gif" alt="Validated source trajectory candidate" width="320" /><br /><sub>Validated source trajectory</sub></td>
    <td width="50%" align="center"><img src="media/gif/foodlike_pilot_outcomes.gif" alt="Controlled pick-and-place pilot outcomes" width="320" /><br /><sub>Controlled target-layout outcomes</sub></td>
  </tr>
</table>

<details>
<summary><strong>Evidence boundary</strong></summary>

Public media illustrate MolmoSpaces simulation rollouts. They do not establish real-robot transfer. CI checks establish repository health; generated HDF5 existence does not alone establish behavioral success. Each workline records its own acceptance boundary and inventory.

</details>

## Worklines and repository map

This repository is a MolmoSpaces integration workline portfolio. The top-level README lists every public workline so a new reader does not need to browse subdirectories to discover what is here.

| Workline | Canonical README | Code entrypoint | Evidence / inventory | Status |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`docs/worklines/mimicgen_pick_and_place/README.md`](docs/worklines/mimicgen_pick_and_place/README.md) | [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](results/workline_index/mimicgen_pick_and_place.md) | Active / primary |
| 50-demo MimicGen cross-subtask route | [`archive/docs/worklines/mimicgen_50cross/README.md`](archive/docs/worklines/mimicgen_50cross/README.md) | [`archive/pnp/legacy_50cross/`](archive/pnp/legacy_50cross/) | [`results/50cross_selectsrc_pilot_20260727_182533/`](results/50cross_selectsrc_pilot_20260727_182533/) | Archived diagnostic |
| Bimanual YAM browser teleoperation | [`docs/worklines/bimanual_yam_browser_teleop/README.md`](docs/worklines/bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`docs/worklines/ithor_bimanual_yam/README.md`](docs/worklines/ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | In progress |
| Completed custom-scene bimanual YAM baseline | [`docs/worklines/bimanual_yam_source_baseline/README.md`](docs/worklines/bimanual_yam_source_baseline/README.md) | Inventory / regeneration entrypoints | [`results/workline_index/bimanual_yam_source_baseline.md`](results/workline_index/bimanual_yam_source_baseline.md) | Completed evidence package |
| MolmoAct2 → MolmoSpaces legacy | [`docs/worklines/molmoact2_legacy/README.md`](docs/worklines/molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](results/workline_index/molmoact2_integration_legacy.md) | Legacy / ended |
| Bounded official MolmoSpaces reproduction | [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) | Upstream MolmoSpaces entrypoints | [`results/workline_index/molmospaces_official_reproduction.md`](results/workline_index/molmospaces_official_reproduction.md) | Completed bounded reproduction |
| MimicGen single-arm groundwork | [`docs/worklines/mimicgen_single_arm/README.md`](docs/worklines/mimicgen_single_arm/README.md) | Superseded by [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_single_arm.md`](results/workline_index/mimicgen_single_arm.md) | Historical groundwork |

Also included: the upstream MolmoSpaces source snapshot, lightweight manifests and summaries under `results/`, small README demo media under `media/`, public documentation, and attribution files.

Large runtime data are excluded from Git: official shards, generated HDF5 files, rollout directories, simulator logs, PID files, caches, videos, local machine paths, and internal planning ledgers. Worklines that use such files keep a README, lightweight inventory, or regeneration entrypoint instead of committing the raw artifact.

## Primary Reproduction Path

The primary workflow in this repository is:

> One validated Franka source demonstration -> MimicGen expansion across independent target layouts -> accepted rollouts

### Mainline: one validated source demonstration to MimicGen expansion

The mainline contract is deliberately small: validate one source demonstration, freeze an independent target-layout manifest, and let MimicGen transform that same source into multiple target layouts. A multi-demo source pool is optional for comparison experiments; it is not a prerequisite for the repository's primary route.

For the generic Franka Pick-and-Place runner, `--mode whole-source` with `--source-count 1` makes the contract explicit: the complete one-demo source is passed to MimicGen for every target entry. The target manifest stays immutable for the run, and each generated rollout is accepted only after replay, persistence, artifact, and uniqueness checks.

```bash
$MOLMOSPACES_PYTHON src/pnp/run_generation.py \
  --work runtime/one_source_expansion \
  --source-hdf5 /path/to/validated_source_one_demo.hdf5 \
  --target-manifest /path/to/target_manifest.json \
  --mode whole-source \
  --source-count 1 \
  --target-success 10 \
  --max-attempts 30 \
  --run-label one_source_mimicgen
```

For the FloorPlan1 bimanual YAM route, the same single-source contract is implemented by the sequential runner: one enriched source demo is converted into right/left MimicGen inputs, both arms run in one physical reset, and each target layout receives a checkpointed result.

```bash
PYTHONPATH=.:vendor/mimicgen:vendor/robomimic $MOLMOSPACES_PYTHON src/pnp_bimanual_yam/run_datagen.py \
  --src /path/to/source_demo_enriched.h5 \
  --num-attempts 16 \
  --sampling-design lhs \
  --xy-jitter 0.10 \
  --manifest runtime/one_source_yam/manifest.json \
  --success-dir runtime/one_source_yam/successes
```

The bimanual command expands one source across sampled target layouts; it does not collect a new source demo per layout.

The command-line value `--robot droid` selects a **Franka robot with DROID-style cameras**. It does not select an RB-Y1 robot. The RB-Y1 CuRobo/planner-server pipeline is a separate optional upstream workline and is not required for the Franka workflow below. See [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) only if that separate workline is your goal.

### Configuration-Driven MimicGen Augmentation

For the reusable source-HDF5, target-manifest, and rollout-generation workflow, use the dedicated engineering guide: [`docs/pnp-data-augmentation.md`](docs/pnp-data-augmentation.md). The supported public boundary is:

```text
configs/pnp/*.json -> scripts/pnp/run_generation.sh -> src/pnp/run_experiment.py -> src/pnp/run_generation.py
```

The configuration records scenario-independent run controls and explicit input artifacts. Shell launchers contain no Python heredocs or experiment-specific paths; historical fixed-layout utilities remain under `archive/pnp/` for provenance only. Start with `configs/pnp/generation.example.json`, run its `--dry-run` validation, then replace its placeholder source HDF5 and target manifest with validated artifacts.

### Current cross-scene target control

The current controlled cross-scene example reuses the replay-verified source pool from
`house 1716` and evaluates it on an independently reset target in `house 3080`
(`procthor-objaverse`, `val` split). The target uses the
`support_adapted_planar_pair` layout, with the Irish potato pickup object and an
explicit place-receptacle UID recorded in the target manifest. One normal simulator
rollout reached final success and remained successful through the post-hold window;
this is a diagnostic control proving the target-layout path, not a formal success-rate
claim, full 6-D rigid migration result, or training-ready dataset.

## Franka Datagen Quick Start

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

SSH also works:

```bash
git clone git@github.com:yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

### 2. Create a Python environment

MolmoSpaces uses Python 3.11.

With conda:

```bash
conda create -n molmospaces-integration python=3.11
conda activate molmospaces-integration
python -m pip install --upgrade pip setuptools wheel
```

With `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

### 3. Install the Franka datagen dependencies

Install MolmoSpaces with the MuJoCo extra. This extra includes HTTPX SOCKS transport, which is harmless without a proxy and prevents the first OpenCLIP download from failing with `socksio` missing when a SOCKS proxy is configured. Franka datagen does not require CuRobo or an RB-Y1 planner server.

```bash
python -m pip install -e ".[mujoco]"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

`pip` may report resolver or `pip check` warnings for optional packages pulled in by the upstream research stack, including missing `rich`, `numba`, `scikit-learn`, `accelerate`, `transformers`, `ninja`, or `py-cpuinfo`, and a platform warning for `decord`. Do not treat those warnings alone as either installation success or failure. The supported Franka gate is the import check below, fixed-pool loading, and a real rollout with validated artifacts. Investigate a warning if one of those gates imports the affected package or fails.

Set persistent cache locations before the first run. `MLSPACES_ASSETS_DIR` stores this checkout's extracted/link farm, LMDB indexes, and generated datagen outputs. `MLSPACES_CACHE_DIR` stores downloaded MolmoSpaces archives and can be shared across checkouts. `HF_HOME` stores model weights, while the NLTK variables store WordNet data.

```bash
export MLSPACES_ASSETS_DIR=${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets/current}
export MLSPACES_CACHE_DIR=${MLSPACES_CACHE_DIR:-$HOME/.cache/molmo-spaces-resources}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
export MOLMOSPACES_NLTK_DATA=$NLTK_DATA
```

The first pool or rollout command may download/extract scenes, objects, and grasps, then print `LMDB: 100%` while building local lookup indexes. The LMDB progress is local indexing, not a model download. A checkout-local asset directory that links to an existing shared resource cache proves fresh code/environment setup, but not a cache-empty machine download.

The commands in this interactive quick start intentionally do not use `set -euo pipefail`; a failed check should report the error without closing a VS Code terminal.

### 4. Verify the base installation

```bash
python - <<'PY'
import mujoco
import torch
import warp
import molmo_spaces

print("MolmoSpaces:", molmo_spaces.__file__)
print("MuJoCo:", mujoco.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.cuda.is_available())
print("Warp CUDA:", warp.is_cuda_available())
PY
```

### 5. Download and verify the NLTK resources

Download WordNet explicitly during setup. Runtime imports only use local resources and fail immediately with this command if either corpus is missing; they do not silently contact `raw.githubusercontent.com`.

```bash
python -m nltk.downloader -d "$NLTK_DATA" wordnet wordnet2022

python - <<'PY'
import nltk

resources = {
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "wordnet2022": ("corpora/wordnet2022", "corpora/wordnet2022.zip"),
}
for name, candidates in resources.items():
    for candidate in candidates:
        try:
            path = nltk.data.find(candidate)
            print(f"NLTK_RESOURCE_OK: {name} -> {path}")
            break
        except LookupError:
            continue
    else:
        raise RuntimeError(f"NLTK_RESOURCE_MISSING: {name}")
PY
```

Do not continue until both resources print `NLTK_RESOURCE_OK`. If the downloader cannot reach GitHub, configure a working network route for this setup step and retry. Later datagen runs do not perform this network check.

### 6. Preflight and cache the OpenCLIP weights

Pick-and-Place task sampling uses `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`. Download its approximately 1.71 GB weight before starting a rollout so a network failure is detected before scene initialization. Start with no proxy and the official endpoint:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
unset HF_ENDPOINT

python - <<'PY'
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

repo_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
filename = "open_clip_pytorch_model.bin"

try:
    metadata = get_hf_file_metadata(hf_hub_url(repo_id, filename), timeout=30)
    print("commit:", metadata.commit_hash)
    print("etag:", metadata.etag)
    print("size:", metadata.size)
    if not metadata.commit_hash or not metadata.etag or not metadata.size:
        print("CLIP_METADATA_INVALID: try the mirror or another network route")
    else:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print("CLIP_WEIGHT:", path)
        print("CLIP_DOWNLOAD_OK")
except Exception as exc:
    print("CLIP_DOWNLOAD_FAILED:", type(exc).__name__, exc)
PY
```

Do not proceed to datagen until the last line is `CLIP_DOWNLOAD_OK`. An HTTP `200` alone is insufficient: `commit`, `etag`, and `size` must all be non-empty. At the time this route was validated, the official response was commit `1627032197142fbe2a7cfec626f4ced3ae60d07a` and size `1710631365`; a future upstream revision may legitimately change them.

If direct access fails, retry the same block after selecting the mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The mirror is a fallback, not a guarantee: large files may still redirect to Hugging Face Xet/CDN hosts. If both direct and mirror routes fail, configure your own network proxy and rerun the metadata check. Never copy a server-specific proxy port from another machine. If a proxy returns `commit: None`, `etag: None`, or `size: None`, disable it or change nodes because it is altering or losing required Hugging Face response headers.

After the download, verify that OpenCLIP can load without any network access. This catches an incomplete snapshot before scene initialization and prevents Hugging Face from probing an uncached `.safetensors` alternative at rollout time.

```bash
HF_HUB_OFFLINE=1 python - <<'PY'
import open_clip

open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="laion2b_s32b_b82k",
    device="cpu",
)
print("CLIP_OFFLINE_LOAD_OK")
PY
```

Do not continue until the final line is `CLIP_OFFLINE_LOAD_OK`.

### 7. List the fixed Franka PnP pools

```bash
python scripts/datagen/run_pipeline.py --list_pools
```

Each pool fixes the scene dataset, split, house, pickup object, and receptacle. A pool remains a research candidate until it passes the full behavior and artifact gates on the target machine.

### 8. Franka datagen option reference

| Option | Meaning and constraints |
|---|---|
| `--pool NAME` | Selects a fixed MolmoData-derived PnP identity. It overrides `--scene_dataset`, `--data_split`, `--house_inds`, `--pickup_obj_name`, and `--fixed_place_receptacle_uid`. It requires `--task_type pick_and_place` and cannot be combined with `--config` or `--eval`. |
| `--samples_per_house N` | Target number of saved trajectories per house. Sampling or IK failures can end a run below this target. Validate the actual success and HDF5 trajectory counts. |
| `--device gpu` | Runs Franka Warp parallel IK on CUDA. MuJoCo physics remains on CPU. `--device cpu` is the slower fallback. |
| `CUDA_VISIBLE_DEVICES=K` | Selects the physical GPU exposed to Warp. `--device gpu` alone does not choose a physical GPU. |
| `--num_workers N` | Sets rollout worker processes. Workers consume independent work items generated by the runner; one work item is never split. Effective parallelism depends on the generated batches, and more workers do not guarantee linear speedup or target completion. |
| `--seed N` | Controls task sampling and randomization. Use a new seed for every continuation run to avoid repeating the same sequence. |
| `--run_name_prefix NAME` | Adds a readable, unique prefix to the timestamped output directory. Use distinct prefixes for separate seeds and pools. |
| `--randomize_fixed_pickup_pose` | Resamples the fixed pickup object on its discovered original supporting geometry, within `--fixed_pickup_min_dist` and `--fixed_pickup_max_dist`. |
| `--filter_for_successful_trajectories` | Saves successful trajectories instead of retaining failures as source candidates. |
| `--disable_action_noise` | Disables per-step robot action noise for controlled source collection. |
| `--require_clean_success` | Sets planner retries to zero and rejects a trajectory if any retry occurs. It requires a supported object-manipulation planner. |
| `--require_success_count N` | Exits nonzero unless the run produces at least `N` successful trajectories. Use this for bounded smoke tests so zero-output runs cannot look successful from the process exit code. |

Keep each pool in a separate output/source dataset. Do not combine identities from different pools and call the result a homogeneous source pool. Outputs are written to `ASSETS_DIR/datagen/<task_type>_<policy>_v1/<prefix>_<timestamp>`; the command below therefore writes under `datagen/pick_and_place_planner_v1/`.

### 9. Run a bounded Franka PnP datagen smoke

On native Linux with an NVIDIA GPU, select a GPU for Warp parallel IK. MuJoCo physics remains on CPU:

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1

python scripts/datagen/run_pipeline.py \
  --robot droid --policy planner --task_type pick_and_place \
  --pool <pool-name> \
  --samples_per_house 1 --randomize_fixed_pickup_pose \
  --filter_for_successful_trajectories \
  --disable_action_noise --require_clean_success \
  --require_success_count 1 \
  --device gpu --num_workers 1 \
  --seed <seed> --run_name_prefix <run-label>
```

Here `--robot droid` means `FrankaRobotConfig` with `FrankaDroidCameraSystem`. Use `--device cpu` only as a slower diagnostic fallback. WSL2 does not expose the NVIDIA EGL device extension required by MolmoSpaces headless rendering, so complete GPU-rendered datagen acceptance requires native Linux with the NVIDIA EGL vendor configuration.

`samples_per_house=1` requests one saved trajectory; it does not prove that a valid demonstration was produced. Accept a datagen run only when the process exits successfully, a non-empty HDF5 trajectory and expected videos are present, control/state arrays are finite, padded arrays are valid under their length or mask fields, the task identity matches the pool, planner phases cover full Pick-and-Place behavior, replay/video shows approach through stable release, and `--require_clean_success` observed no planner retry. `--require_success_count 1` makes a zero-success smoke return a nonzero exit code, but artifact and behavior checks are still required.

Generated files are written under the MolmoSpaces resource datagen directory printed by the run. Config construction, scene loading, or an HDF5 file alone is not datagen success.

## Continue to MimicGen

Complete the Franka datagen and artifact gates before using its HDF5 as MimicGen source input.

### 1. Fetch pinned MimicGen and robomimic dependencies

`vendor/` is intentionally not committed to Git. Fetch the upstream repositories at the commits used by this workline:

```bash
bash tools/setup_mimicgen_dependency.sh
```

The script currently pins:

- MimicGen: `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb`
- robomimic: `e10526b9a40c78b41f1e37e60041dc0ec0a5f60f`

Install them in editable mode:

```bash
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

Set dependency roots and `PYTHONPATH`:

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
export PYTHONPATH=$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT:${PYTHONPATH:-}
```

If existing local checkouts are used instead, point `MIMICGEN_ROOT` and `ROBOMIMIC_ROOT` to those directories.

### 2. Set runtime paths

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
export MOLMOSPACES_NLTK_DATA=$NLTK_DATA

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

`HF_HOME` is used by robomimic's CLIP language embedding utility. `NLTK_DATA` / `MOLMOSPACES_NLTK_DATA` are useful when MolmoSpaces needs a local WordNet cache.

### 3. Active Pick-and-Place pipeline

The active source build and generation path is parameterized; source count, source HDF5, target manifest, target range, and output label are run arguments rather than script names. The mainline uses one validated source demonstration; the 17-demo replay-verified pilot is retained as an optional multi-source comparison.

```bash
$MOLMOSPACES_PYTHON src/pnp/run_source_hdf5_pipeline.py --help
$MOLMOSPACES_PYTHON src/pnp/sample_fixedbase_target_manifest.py --help
$MOLMOSPACES_PYTHON src/pnp/run_generation.py --help
```

Build or select one validated source HDF5, create and validate an independent target manifest, then run the one-source command above. `generate_pick_place_rollout.py` is the single-rollout execution primitive. Multi-source `per-subtask` selection remains available for comparison studies. Full usage and evidence gates are in [`src/pnp/README.md`](src/pnp/README.md).

Historical 50-demo cross-subtask scripts and collectors are archived under [`archive/pnp/`](archive/pnp/); the corresponding result directories remain unchanged under `results/`.

## Bimanual YAM Browser Teleoperation

Browser-based keyboard teleoperation for bimanual YAM scenes. The supported public entrypoint is the keyboard teleoperation bridge, which performs the current strict tabletop initialization before serving the browser UI. The older read-only viewer script is not the recommended reproduction entrypoint for this workline.

### Environment requirements

- **GPU rendering is required** for usable frame rates. CPU software rendering (OSMesa) with three cameras gives ~3 FPS, which is too slow for responsive keyboard control.
- **NVIDIA EGL headless rendering** is the tested configuration: a Linux host with an NVIDIA GPU, the NVIDIA proprietary driver, and EGL vendor files (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`). The teleop automatically uses `MUJOCO_GL=egl` via MolmoSpaces.
- **WSL2 is not supported** for this workline. WSL2 uses Mesa EGL, which does not expose `EGL_EXT_platform_device` — the extension MolmoSpaces relies on for headless GPU rendering. If you are on WSL2, run the teleop on a remote Linux GPU server instead (see below).

### Running locally (Linux + NVIDIA GPU)

Run the teleoperation bridge:

```bash
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
$MOLMOSPACES_PYTHON src/bimanual_yam/browser_keyboard_teleop.py \
  --host 127.0.0.1 \
  --port 8765 \
  --house-index 1 \
  --seed 110 \
  --render-fps 8 \
  --control-hz 25 \
  --input-timeout-ms 400 \
  --initialization-max-attempts 50 \
  --initialization-report runtime/bimanual_yam_initialization_report.json
```

If the CLIP model is cached elsewhere, set `HF_HOME` to that directory. The MuJoCo extra in step 3 installs HTTPX SOCKS transport; a `socksio` error indicates that this checkout was installed without the current extra and should be reinstalled.

Open `http://127.0.0.1:8765` after the terminal prints the local teleoperation URL.

### Running on a remote GPU server

If you run the teleop on a remote Linux GPU server, set up an SSH tunnel to forward the browser port to your local machine:

```bash
# On your local machine:
ssh -L 8765:127.0.0.1:8765 user@your-gpu-server
```

Then open `http://127.0.0.1:8765` in your local browser.

### Browser controls

Click the page first; `Tab` switches active arm; `W/S/A/D` moves in the visual plane; `E/Q` raises/lowers; arrow keys control pitch/yaw; `Z/C` roll; `F` toggles the active gripper. If initialization fails, increase `--initialization-max-attempts` or inspect `runtime/bimanual_yam_initialization_report.json`.

## Included Results Snapshot

- **Heterogeneous Pick-and-Place whole-source generation**: `10/10` accepted generated rollouts with full rollout, final success, persistent success, and 30-step post-hold. See `results/whole_source_transformfirst_summary.json`.
- **Homogeneous foodlike-to-bowl pilot**: strict automatic success `13/100`; reviewed visual success `15/100` after two one-frame trace-glitch cases.
- **50-demo cross-subtask source pool**: `51` strict replay/datagen-info hard-pass candidates were screened, `50` were selected into `robomimic_pnp_50demo_crossmix_aligned.hdf5`, with `9286` total source action rows. See `results/pnp_50cross_selected_hardpass_indices.json` and `results/robomimic_pnp_50demo_crossmix_aligned.summary.json`.
- **Broad random `select_src_per_subtask=True` pilot**: first completed samples exposed geometry/contact/source-compatibility issues. Lightweight logs and generation traces are under `results/50cross_selectsrc_pilot_20260727_182533/`; large binary videos/HDF5/arrays are excluded from Git.
- **Uniform collector live snapshot**: non-deduplicated progress in `results/collector_uniform_summary_live.json`.
- **High-yield deduplicated collector live snapshot**: unique accepted progress in `results/collector_highyield_dedup_summary_live.json`.

<details>
<summary>Demo media</summary>

### Generated Rollout Examples

[![Generated rollout examples](media/gif/heterogeneous_generated_examples.gif)](media/heterogeneous_generated_examples.mp4)

[Open MP4](media/heterogeneous_generated_examples.mp4)

### Source Candidates

[![Foodlike source candidates](media/gif/foodlike_source_candidates.gif)](media/foodlike_source_candidates_grid.mp4)

[Open MP4](media/foodlike_source_candidates_grid.mp4)

### Pilot Outcomes

[![Foodlike pilot outcomes](media/gif/foodlike_pilot_outcomes.gif)](media/foodlike_pilot_outcomes.mp4)

[Open MP4](media/foodlike_pilot_outcomes.mp4)

</details>

## Repository Layout

```text
molmo_spaces/             Upstream MolmoSpaces Python package
scripts/                  Upstream MolmoSpaces scripts
configs/, examples/, docs/ Upstream configuration, examples, and documentation
src/pnp/                  Pick-and-Place MimicGen integration scripts
src/bimanual_yam/         Bimanual YAM diagnostics and browser keyboard teleoperation
results/                  Lightweight JSON manifests and result summaries
media/                    Small public demo videos for the GitHub README
docs/experiments.md       Public experiment notes and evidence boundaries
docs/upstream_molmospaces_readme.md  Original upstream MolmoSpaces README
tools/setup_mimicgen_dependency.sh   Helper to fetch MimicGen and robomimic into vendor/
```

## Evidence Boundaries

- Source trajectories are synthetic planner expert trajectories from MolmoBot-Data, not human demonstrations.
- Replay success and parser success are prerequisites, not generated-demo success.
- Accepted generated demonstrations require a real MolmoSpaces simulator rollout, `final_success=true`, post-hold stability, and saved artifacts.

## Attribution

- Upstream MolmoSpaces: Allen Institute for AI, Apache License 2.0. The upstream source and license are retained in this repository.
- Upstream MimicGen: NVIDIA / NVlabs. MimicGen was introduced by Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. MimicGen code is released under the NVIDIA Source Code License, and MimicGen datasets are released under CC-BY 4.0.
- MolmoSpaces Integration and release organization: Kunyu Yang, Institute of Trustworthy Embodied Intelligence, Fudan University.
- Details: `AUTHORS.md`.

## License

The upstream MolmoSpaces code is distributed under the Apache License 2.0; see `LICENSE`. Third-party dependencies and datasets retain their own licenses. The integration code in this repository is provided as research code under the same repository license unless a file states otherwise.
