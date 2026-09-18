from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(os.environ.get("MOLMOSPACES_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT))

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.configs.policy_configs import DummyPolicyConfig
from molmo_spaces.policy.dummy_policy import DummyPolicy
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
from src.pnp.table_support import check_table_support


class EvalCfg(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DummyPolicyConfig = DummyPolicyConfig(policy_cls=DummyPolicy)
    task_horizon: int = 700


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args()
    d = json.loads(a.manifest.read_text())
    rows = d["seeds"]
    indices = range(len(rows)) if a.index < 0 else [a.index]
    reports = []
    for i in indices:
        row = rows[i]
        spec = EpisodeSpec.model_validate(row["episode_spec"])
        sampler = JsonEvalTaskSampler(exp_config=EvalCfg(), episode_spec=spec)
        try:
            task = sampler.sample_task(house_index=int(row["house_id"]))
            task.reset()
            task._env.step(10)
            name = spec.task["pickup_obj_name"]
            r = check_table_support(task._env.current_data, name, a.margin)
            reports.append({"index": i, **r.__dict__})
        finally:
            sampler.close()
    print(
        json.dumps(
            {"valid": all(x["valid"] for x in reports), "reports": reports},
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )
    )
    raise SystemExit(0 if all(x["valid"] for x in reports) else 1)


if __name__ == "__main__":
    main()
