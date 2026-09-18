<div align="center">
  <img src="media/banner.png" alt="MolmoSpaces x MimicGen: Franka pick-and-place trajectory augmentation" width="980" />
</div>

<p align="center">
  <a href="README.md">English</a> &nbsp;|&nbsp; <a href="README_zh.md"><strong>中文</strong></a>
</p>

<h1 align="center">MolmoSpaces x MimicGen</h1>

<p align="center">
  <strong>面向仿真的受控 Pick-and-Place 轨迹扩充。</strong><br />
  从通过验证的 Franka 源示范到变换后的 target 布局，并保留明确的 replay、持久化与唯一性验收门。
</p>

<p align="center">
  <code>Franka + DROID cameras</code> &nbsp; <code>whole-source / per-subtask transfer</code> &nbsp; <code>严格 artifact 验收</code>
</p>

<p align="center">
  <img src="media/gif/heterogeneous_generated_examples.gif" alt="Generated Franka pick-and-place rollouts" width="720" />
</p>

<p align="center">
  <a href="media/heterogeneous_generated_examples.mp4">查看完整尺寸的生成 rollout</a>
</p>

## 主工作线

```text
一条通过验证的 source demonstration  ->  跨 target 布局的 MimicGen 扩充  ->  replay、视频、HDF5 与唯一性检查
```

仓库聚焦可复现的 Franka Pick-and-Place 路线。它将源示范采集、target 布局采样和生成 rollout 的验收分开记录，避免将可运行进程、已保存 artifact 与严格任务成功混为同一层证据。

<table>
  <tr>
    <td width="50%" align="center"><img src="media/gif/source_candidate_example.gif" alt="Validated source trajectory candidate" width="320" /><br /><sub>通过验证的源轨迹</sub></td>
    <td width="50%" align="center"><img src="media/gif/foodlike_pilot_outcomes.gif" alt="Controlled pick-and-place pilot outcomes" width="320" /><br /><sub>受控 target-layout 结果</sub></td>
  </tr>
</table>

<details>
<summary><strong>证据边界</strong></summary>

公开媒体展示的是 MolmoSpaces 仿真 rollout，不构成真实机器人迁移结论。CI 只证明仓库健康；生成 HDF5 的存在也不单独证明行为成功。每条 workline 均在其文档中记录各自的验收边界和 inventory。

</details>

## 工作线与仓库地图

本仓库是 MolmoSpaces 集成工作线合集。首页直接列出全部公开 workline，读者无需翻子目录即可了解仓库包含的所有方向。

| 工作线 | 详细 README | 代码入口 | 证据 / inventory | 状态 |
|---|---|---|---|---|
| MimicGen Pick-and-Place | [`docs/worklines/mimicgen_pick_and_place/README.md`](docs/worklines/mimicgen_pick_and_place/README.md) | [`src/pnp/`](src/pnp/) | [`results/workline_index/mimicgen_pick_and_place.md`](results/workline_index/mimicgen_pick_and_place.md) | Active / primary |
| 50-demo MimicGen cross-subtask route | [`archive/docs/worklines/mimicgen_50cross/README.md`](archive/docs/worklines/mimicgen_50cross/README.md) | [`archive/pnp/legacy_50cross/`](archive/pnp/legacy_50cross/) | [`results/50cross_selectsrc_pilot_20260727_182533/`](results/50cross_selectsrc_pilot_20260727_182533/) | Archived diagnostic |
| Bimanual YAM browser teleop | [`docs/worklines/bimanual_yam_browser_teleop/README.md`](docs/worklines/bimanual_yam_browser_teleop/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | Infrastructure |
| iTHOR bimanual YAM | [`docs/worklines/ithor_bimanual_yam/README.md`](docs/worklines/ithor_bimanual_yam/README.md) | [`src/bimanual_yam/`](src/bimanual_yam/) | [`results/workline_index/ithor_bimanual_yam.md`](results/workline_index/ithor_bimanual_yam.md) | In progress |
| Custom-scene bimanual YAM baseline | [`docs/worklines/bimanual_yam_source_baseline/README.md`](docs/worklines/bimanual_yam_source_baseline/README.md) | Inventory / regeneration entrypoints | [`results/workline_index/bimanual_yam_source_baseline.md`](results/workline_index/bimanual_yam_source_baseline.md) | Completed |
| MolmoAct2 → MolmoSpaces legacy | [`docs/worklines/molmoact2_legacy/README.md`](docs/worklines/molmoact2_legacy/README.md) | [`src/molmoact2_legacy/`](src/molmoact2_legacy/) | [`results/workline_index/molmoact2_integration_legacy.md`](results/workline_index/molmoact2_integration_legacy.md) | Legacy |
| MolmoSpaces bounded reproduction | [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md) | Upstream MolmoSpaces entrypoints | [`results/workline_index/molmospaces_official_reproduction.md`](results/workline_index/molmospaces_official_reproduction.md) | Completed |
| MimicGen single-arm groundwork | [`docs/worklines/mimicgen_single_arm/README.md`](docs/worklines/mimicgen_single_arm/README.md) | 已由 [`src/pnp/`](src/pnp/) 继承 | [`results/workline_index/mimicgen_single_arm.md`](results/workline_index/mimicgen_single_arm.md) | Historical |

同时包含：上游 MolmoSpaces 源码快照、`results/` 下的轻量 manifest 和 summary、`media/` 下的演示媒体、公开文档和 attribution 文件。

大体积运行数据不提交到 Git：官方 shards、生成的 HDF5、rollout 目录、simulator logs、PID 文件、cache、视频、本地机器路径和内部 planning ledger。使用此类文件的 workline 会在仓库中保留 README、轻量 inventory 或再生成入口，而非提交原始产物。

## 主复现路线

本仓库的主流程是：

> 一条通过验证的 Franka source demonstration -> MimicGen 扩充到多个独立 target 布局 -> 验收生成 rollout

### 主线：一条 source demonstration 经过 MimicGen 扩充

主线流程刻意保持最小闭环：先验证一条 source demonstration，再冻结独立的 target-layout manifest，最后让 MimicGen 将同一条 source 变换到多个 target 布局。多条 source 只用于对比实验，不是主线的前置条件。

对通用 Franka Pick-and-Place runner，使用 `--mode whole-source` 加 `--source-count 1` 明确表达这一契约：每个 target entry 都接收完整的单条 source。一次生成运行期间 target manifest 保持不可变；每条 rollout 仍需通过 replay、持久化、artifact 和唯一性检查。

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

FloorPlan1 双臂 YAM 路线也遵循同一条 single-source 契约：一条 enriched source demo 被转换成右/左臂 MimicGen 输入，两只手臂在同一次物理 reset 中顺序执行，每个 target 布局写入 checkpoint 结果。

```bash
PYTHONPATH=.:vendor/mimicgen:vendor/robomimic $MOLMOSPACES_PYTHON src/pnp_bimanual_yam/run_datagen.py \
  --src /path/to/source_demo_enriched.h5 \
  --num-attempts 16 \
  --sampling-design lhs \
  --xy-jitter 0.10 \
  --manifest runtime/one_source_yam/manifest.json \
  --success-dir runtime/one_source_yam/successes
```

这条双臂命令是把一条 source 扩充到采样得到的多个 target 布局，不是为每个布局重新采集 source。

命令行参数 `--robot droid` 表示 **Franka 机器人搭配 DROID 风格相机**，不是 RB-Y1。RB-Y1 的 CuRobo/planner-server pipeline 是另一条可选上游工作线，不是下面 Franka 主流程的依赖。只有确实需要 RB-Y1 时，才阅读 [`docs/worklines/molmospaces_official_reproduction/README.md`](docs/worklines/molmospaces_official_reproduction/README.md)。

## Franka Datagen Quick Start

### 1. Clone

```bash
git clone https://github.com/yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

也可以使用 SSH：

```bash
git clone git@github.com:yanggoumao2/molmospaces-integration.git
cd molmospaces-integration
```

### 2. 创建 Python 环境

MolmoSpaces 使用 Python 3.11。

使用 conda：

```bash
conda create -n molmospaces-integration python=3.11
conda activate molmospaces-integration
python -m pip install --upgrade pip setuptools wheel
```

使用 `uv`：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

### 3. 安装 Franka datagen 依赖

安装 MolmoSpaces 的 MuJoCo extra。该 extra 已包含 HTTPX SOCKS transport；即使不使用代理也没有副作用，使用 SOCKS proxy 时则可避免首次下载 OpenCLIP 因缺少 `socksio` 而失败。Franka datagen 不需要 CuRobo，也不需要 RB-Y1 planner server。

```bash
python -m pip install -e ".[mujoco]"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

`pip` 或 `pip check` 可能报告 upstream research stack 引入的可选包告警，例如缺少 `rich`、`numba`、`scikit-learn`、`accelerate`、`transformers`、`ninja`、`py-cpuinfo`，以及 `decord` 的平台告警。不能仅凭这些告警判断安装成功或失败。Franka 主线的支持 gate 是下面的 import check、固定 pool 加载，以及带完整产物验收的真实 rollout。如果某个 gate 导入了对应包或失败，再调查相关告警。

首次运行前设置持久 cache 路径。`MLSPACES_ASSETS_DIR` 保存当前 checkout 的解压/链接目录、LMDB 索引和 datagen 输出；`MLSPACES_CACHE_DIR` 保存下载的 MolmoSpaces archives，可供多个 checkout 共享；`HF_HOME` 保存模型权重；NLTK 变量保存 WordNet 数据。

```bash
export MLSPACES_ASSETS_DIR=${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets/current}
export MLSPACES_CACHE_DIR=${MLSPACES_CACHE_DIR:-$HOME/.cache/molmo-spaces-resources}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
export MOLMOSPACES_NLTK_DATA=$NLTK_DATA
```

第一次执行 pool 或 rollout 命令时，程序可能下载/解压 scenes、objects 和 grasps，随后打印 `LMDB: 100%` 来建立本地查询索引。LMDB 进度是本地建索引，不是模型下载。checkout 独立的 asset 目录如果链接到已有共享资源 cache，只能证明 fresh code/environment setup，不能证明 cache-empty 新机器下载已通过。

本交互式 Quick Start 的命令有意不使用 `set -euo pipefail`；检查失败时应报告错误，不应关闭 VS Code terminal。

### 4. 验证基础安装

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

### 5. 下载并验证 NLTK 资源

在 setup 阶段显式下载 WordNet。运行期 import 只使用本地资源；任一 corpus 缺失时会立即给出下面的准备命令，不会静默访问 `raw.githubusercontent.com`。

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

只有两个资源都打印 `NLTK_RESOURCE_OK` 才能继续。如果 downloader 无法连接 GitHub，应为该 setup 步骤配置可用网络路径后重试；后续 datagen 不会再做这次联网检查。

### 6. 预检并缓存 OpenCLIP 权重

Pick-and-Place task sampling 使用 `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`。在启动 rollout 前先下载约 1.71 GB 权重，避免场景初始化后才暴露网络失败。默认先关闭代理并使用官方 endpoint 直连：

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
        print("CLIP_METADATA_INVALID: 请尝试镜像或其他网络路径")
    else:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print("CLIP_WEIGHT:", path)
        print("CLIP_DOWNLOAD_OK")
except Exception as exc:
    print("CLIP_DOWNLOAD_FAILED:", type(exc).__name__, exc)
PY
```

只有最后一行出现 `CLIP_DOWNLOAD_OK` 才能继续 datagen。仅看到 HTTP `200` 不够，`commit`、`etag` 和 `size` 必须都非空。本路线验收时官方响应的 commit 为 `1627032197142fbe2a7cfec626f4ced3ae60d07a`、size 为 `1710631365`；未来 upstream revision 可能合理变化。

如果直连失败，设置镜像 endpoint 后重新执行同一 Python block：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

镜像只是 fallback，不保证一定成功：大文件仍可能跳转到 Hugging Face Xet/CDN hosts。如果直连和镜像都失败，再配置用户自己的网络代理并重新执行 metadata check。不要照抄其他服务器的专用代理端口。如果代理下出现 `commit: None`、`etag: None` 或 `size: None`，说明代理丢失或改写了 Hugging Face 必需的响应头，应关闭代理或更换节点。

下载完成后验证 OpenCLIP 能在完全离线时加载。这个 gate 可以在 scene 初始化前发现不完整 snapshot，并避免 rollout 时 Hugging Face 再探测未缓存的 `.safetensors` 候选。

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

只有最后一行出现 `CLIP_OFFLINE_LOAD_OK` 才能继续。

### 7. 查看固定 Franka PnP pools

```bash
python scripts/datagen/run_pipeline.py --list_pools
```

每个 pool 固定 scene dataset、split、house、pickup object 和 receptacle。在目标机器通过完整行为与产物 gate 之前，这些 pool 仍是研究候选。

### 8. Franka datagen 参数参考

| 参数 | 含义与约束 |
|---|---|
| `--pool NAME` | 选择固定的 MolmoData-derived PnP identity。它会覆盖 `--scene_dataset`、`--data_split`、`--house_inds`、`--pickup_obj_name` 和 `--fixed_place_receptacle_uid`；要求 `--task_type pick_and_place`，且不能与 `--config` 或 `--eval` 同时使用。 |
| `--samples_per_house N` | 每个 house 目标保存的轨迹数。Sampling 或 IK 失败可能让 run 提前结束，因此必须核验实际 success count 和 HDF5 trajectory count。 |
| `--device gpu` | 让 Franka Warp parallel IK 使用 CUDA；MuJoCo physics 仍在 CPU。`--device cpu` 是较慢的 fallback。 |
| `CUDA_VISIBLE_DEVICES=K` | 选择 Warp 可见的物理 GPU。仅设置 `--device gpu` 不负责选择具体物理 GPU。 |
| `--num_workers N` | 设置 rollout worker 进程数。Worker 消费 runner 生成的独立 work item；一个 work item 不会被拆分。实际并行度取决于生成的 batches，增加 worker 不保证线性加速或一定完成目标数。 |
| `--seed N` | 控制 task sampling 和 randomization。续采时每个 run 使用新 seed，避免重复相同随机序列。 |
| `--run_name_prefix NAME` | 为时间戳输出目录增加可读且唯一的前缀。不同 seed 和 pool 使用不同 prefix。 |
| `--randomize_fixed_pickup_pose` | 在已发现的原始 supporting geometry 上重采样固定 pickup，距离范围由 `--fixed_pickup_min_dist` 和 `--fixed_pickup_max_dist` 控制。 |
| `--filter_for_successful_trajectories` | 只把成功轨迹保存为 source candidate，不保留失败轨迹。 |
| `--disable_action_noise` | 关闭逐步 robot action noise，用于受控 source collection。 |
| `--require_clean_success` | 将 planner retry 设为零，并拒绝发生任何 retry 的轨迹；要求支持该字段的 object-manipulation planner。 |
| `--require_success_count N` | 如果成功轨迹少于 `N`，进程以非零状态退出。Bounded smoke 应使用该参数，避免零产物运行仅凭退出码看起来成功。 |

每个 pool 应使用独立的 output/source dataset。不要混合不同 pool 的 identity 后把结果称为 homogeneous source pool。输出目录是 `ASSETS_DIR/datagen/<task_type>_<policy>_v1/<prefix>_<timestamp>`；所以下面的命令输出到 `datagen/pick_and_place_planner_v1/` 下。

### 9. 运行有界 Franka PnP datagen smoke

在原生 Linux NVIDIA 机器上，为 Warp parallel IK 选择 GPU；MuJoCo physics 仍在 CPU：

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

这里 `--robot droid` 实际使用 `FrankaRobotConfig` 和 `FrankaDroidCameraSystem`。`--device cpu` 只建议作为较慢的诊断 fallback。WSL2 不提供 MolmoSpaces headless rendering 所需的 NVIDIA EGL device extension，因此完整 GPU 渲染 datagen 验收需要带 NVIDIA EGL vendor 配置的原生 Linux。

`samples_per_house=1` 只是请求保存一条轨迹，不证明已经生成有效 demo。只有进程正常退出、输出目录包含非空 HDF5 和预期视频、控制/状态数组 finite、padding 数组符合其有效长度或 mask 语义、task identity 与 pool 一致、planner phases 覆盖完整行为、视频/replay 显示 approach 到稳定 release，并且没有 planner retry，才能验收 datagen。`--require_success_count 1` 会让零成功 smoke 返回非零退出码，但仍不能替代产物与行为检查。

生成文件位于运行日志打印的 MolmoSpaces resource datagen 目录。仅有 config 构造、scene load 或 HDF5 文件不能证明 datagen 成功。

## 继续进入 MimicGen

先完成 Franka datagen 与产物验收，再把 HDF5 作为 MimicGen source。

### 1. 拉取固定版本的 MimicGen 和 robomimic

`vendor/` 不直接提交到 Git。请用脚本拉取本工作线实际使用的 upstream commit：

```bash
bash tools/setup_mimicgen_dependency.sh
```

当前 pin：

- MimicGen: `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb`
- robomimic: `e10526b9a40c78b41f1e37e60041dc0ec0a5f60f`

editable 安装：

```bash
pip install -e vendor/robomimic
pip install -e vendor/mimicgen
```

设置依赖路径和 `PYTHONPATH`：

```bash
export MIMICGEN_ROOT=$PWD/vendor/mimicgen
export ROBOMIMIC_ROOT=$PWD/vendor/robomimic
export PYTHONPATH=$PWD:$MIMICGEN_ROOT:$ROBOMIMIC_ROOT:${PYTHONPATH:-}
```

如果使用已有本地 checkout，把 `MIMICGEN_ROOT` 和 `ROBOMIMIC_ROOT` 指向对应目录即可。

### 2. 设置运行路径

```bash
export MOLMOSPACES_ROOT=$PWD
export MOLMOSPACES_PYTHON=python
export MOLMOSPACES_PNP_WORKDIR=$PWD/runtime/mimicgen_pick_and_place
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
export MOLMOSPACES_NLTK_DATA=$NLTK_DATA

mkdir -p "$MOLMOSPACES_PNP_WORKDIR"/{artifacts/seeds,artifacts/mimicgen_pnp,data/molmobot_data/FrankaPickAndPlaceOmniCamConfig/val_shards,logs}
```

`HF_HOME` 会被 robomimic 的 CLIP language embedding 工具使用。`NLTK_DATA` / `MOLMOSPACES_NLTK_DATA` 用于 MolmoSpaces 需要本地 WordNet cache 的情况。

### 3. 当前 Pick-and-Place pipeline

当前 source 构建和生成路径均为参数化 CLI；source 数量、source HDF5、target manifest、target 范围和输出标签均通过运行参数提供，不再编码在脚本名中。主线使用一条通过验证的 source demonstration；17 条唯一、回放验证通过的 source pilot 保留为可选的多 source 对比。

```bash
$MOLMOSPACES_PYTHON src/pnp/run_source_hdf5_pipeline.py --help
$MOLMOSPACES_PYTHON src/pnp/sample_fixedbase_target_manifest.py --help
$MOLMOSPACES_PYTHON src/pnp/run_generation.py --help
```

先构建或选择一条通过验证的 source HDF5，再创建并验证独立 target manifest，之后执行上面的 single-source 命令。`generate_pick_place_rollout.py` 是单条真实 simulator rollout 的执行原语；多 source 的 `per-subtask` selection 保留给对比实验。完整用法和证据 gate 见 [`src/pnp/README.md`](src/pnp/README.md)。


### 当前跨场景 target control

当前受控跨场景示例复用通过回放验证的 `house 1716` source pool，在独立 reset 的
`house 3080` target（`procthor-objaverse` 的 `val` split）上进行评估。该 target
使用 `support_adapted_planar_pair` 布局，pickup object 为 Irish potato，放置容器
UID、target layout 和其他 reset 状态均记录在 target manifest 中。一次正常 simulator
rollout 达到 final success，并在 post-hold 窗口内保持成功；这只是证明 target-layout
路径可运行的 diagnostic control，不是正式成功率、完整 6-D 刚体迁移或可直接训练数据的声明。

历史 50-demo cross-subtask 脚本与旧 collector 已移入 [`archive/pnp/`](archive/pnp/)，对应的 `results/` 结果目录保持原路径不变。

## Bimanual YAM 浏览器遥操作

bimanual YAM 场景的浏览器键盘遥操作工具。公开复现推荐入口是键盘遥操作 bridge，它会先执行当前 strict tabletop 初始化，再启动浏览器 UI。旧的只读 viewer 脚本不是此工作线推荐复现入口。

### 运行环境要求

- **需要 GPU 渲染**才能达到可用帧率。CPU 软件渲染（OSMesa）三相机模式下仅 ~3 FPS，键盘控制响应延迟严重。
- **NVIDIA EGL headless 渲染**是已验证的配置：Linux 主机 + NVIDIA GPU + NVIDIA 闭源驱动 + EGL vendor 文件（`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`）。遥操作工具通过 MolmoSpaces 自动使用 `MUJOCO_GL=egl`。
- **WSL2 不支持**此工作线。WSL2 使用 Mesa EGL，不暴露 `EGL_EXT_platform_device` 扩展——这是 MolmoSpaces headless GPU 渲染依赖的扩展。如果使用 WSL2，请在远程 Linux GPU 服务器上运行（见下方说明）。

### 本地运行（Linux + NVIDIA GPU）

启动遥操作 bridge：

```bash
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

终端打印本地 teleoperation URL 后打开 `http://127.0.0.1:8765`。

### 在远程 GPU 服务器上运行

如果 teleop 运行在远程 Linux GPU 服务器上，需要建立 SSH tunnel 将浏览器端口转发到本地机器：

```bash
# 在本地机器上执行：
ssh -L 8765:127.0.0.1:8765 user@your-gpu-server
```

然后在本地浏览器打开 `http://127.0.0.1:8765`。

### 浏览器键位

先点击页面；`Tab` 切换 active arm；`W/S/A/D` 在视觉平面移动；`E/Q` 上下；方向键控制 pitch/yaw；`Z/C` roll；`F` 切换当前夹爪。如果初始化失败，增大 `--initialization-max-attempts` 或查看 `runtime/bimanual_yam_initialization_report.json`。

## 当前结果快照

- **Heterogeneous Pick-and-Place whole-source generation**：`10/10` accepted generated rollouts，均满足 full rollout、final success、persistent success 和 30-step post-hold。见 `results/whole_source_transformfirst_summary.json`。
- **Homogeneous foodlike-to-bowl pilot**：严格自动成功 `13/100`；人工复核视觉成功 `15/100`，包含两个 one-frame trace-glitch case。
- **50-demo cross-subtask source pool**：`51` 个严格回放/datagen-info hard-pass 候选被筛选，`50` 个选入 `robomimic_pnp_50demo_crossmix_aligned.hdf5`，共 `9286` source action rows。见 `results/pnp_50cross_selected_hardpass_indices.json` 和 `results/robomimic_pnp_50demo_crossmix_aligned.summary.json`。
- **Broad random `select_src_per_subtask=True` pilot**：首批样本暴露了 geometry/contact/source-compatibility 问题。轻量日志和生成轨迹见 `results/50cross_selectsrc_pilot_20260727_182533/`，大体积视频/HDF5/数组不在 Git 中。
- **Uniform collector live snapshot**：见 `results/collector_uniform_summary_live.json`，未去重。
- **High-yield dedup collector live snapshot**：见 `results/collector_highyield_dedup_summary_live.json`，按 action hash 去重。

<details>
<summary>演示媒体</summary>

### 生成轨迹示例

[![Generated rollout examples](media/gif/heterogeneous_generated_examples.gif)](media/heterogeneous_generated_examples.mp4)

[打开 MP4](media/heterogeneous_generated_examples.mp4)

### 源轨迹候选

[![Foodlike source candidates](media/gif/foodlike_source_candidates.gif)](media/foodlike_source_candidates_grid.mp4)

[打开 MP4](media/foodlike_source_candidates_grid.mp4)

### Pilot 结果

[![Foodlike pilot outcomes](media/gif/foodlike_pilot_outcomes.gif)](media/foodlike_pilot_outcomes.mp4)

[打开 MP4](media/foodlike_pilot_outcomes.mp4)

</details>

## 仓库结构

```text
molmo_spaces/             上游 MolmoSpaces Python 包
scripts/                  上游 MolmoSpaces 脚本
configs/, examples/, docs/ 上游配置、示例和文档
src/pnp/                  Pick-and-Place MimicGen 集成脚本
src/bimanual_yam/         bimanual YAM 诊断、浏览器可视化和键盘遥操作
results/                  轻量 JSON manifest 和结果摘要
media/                    GitHub README 使用的小体积演示视频
docs/experiments.md       实验流程、结果边界和证据说明
docs/upstream_molmospaces_readme.md  上游 MolmoSpaces 原始 README
tools/setup_mimicgen_dependency.sh   拉取 MimicGen 和 robomimic 到 vendor/ 的辅助脚本
```

## 证据边界

- Source trajectories 是 MolmoBot-Data 中的 synthetic planner expert trajectories，不是 human demonstrations。
- Replay success 和 parser success 是前置条件，不等于 generated-demo success。
- Accepted generated demonstrations 需要真实 MolmoSpaces simulator rollout、`final_success=true`、post-hold stability 和保存的 artifacts。

## 作者与引用

- 上游 MolmoSpaces：Allen Institute for AI，Apache License 2.0。上游源码和 license 已保留在本仓库中。
- 上游 MimicGen：NVIDIA / NVlabs。MimicGen 论文作者为 Ajay Mandlekar、Soroush Nasiriany、Bowen Wen、Iretiayo Akinola、Yashraj Narang、Linxi Fan、Yuke Zhu 和 Dieter Fox。MimicGen 代码遵循 NVIDIA Source Code License，MimicGen 数据集遵循 CC-BY 4.0。
- MolmoSpaces Integration：Kunyu Yang，复旦大学可信具身智能研究院。
- 详细说明见 `AUTHORS.md`。

## License

上游 MolmoSpaces 代码遵循 Apache License 2.0；见 `LICENSE`。第三方依赖和数据集保留各自 license。本仓库中的集成代码按本仓库 license 作为 research code 发布，除非文件另有说明。
