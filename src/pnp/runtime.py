"""Shared runtime helpers for parameterized MolmoSpaces workflows."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or base is None:
        return candidate.resolve()
    return (base / candidate).resolve()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object: {path}")
            rows.append(value)
    return rows


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pythonpath(root: Path, current: str | None = None) -> str:
    entries = [root, root / "vendor/mimicgen", root / "vendor/robomimic"]
    if current:
        entries.append(Path(current))
    return os.pathsep.join(str(path) for path in entries if str(path))


def workflow_env(
    root: Path, work: Path, *, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = dict(base or os.environ)
    env["MOLMOSPACES_ROOT"] = str(root)
    env["MOLMOSPACES_PNP_WORKDIR"] = str(work)
    env["PYTHONPATH"] = build_pythonpath(root, env.get("PYTHONPATH"))
    return env


def run_logged(
    command: Sequence[str],
    log: Path,
    env: Mapping[str, str],
    *,
    cwd: Path | None = None,
    timeout_sec: int | None = None,
) -> int:
    """Run one rollout in its own process group with an optional hard bound."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(env),
            cwd=cwd,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None)
        except subprocess.TimeoutExpired:
            stream.write(
                f"\nRUNNER_TIMEOUT: rollout exceeded {timeout_sec}s; terminating process group\n"
            )
            stream.flush()
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124
        return int(process.returncode)
