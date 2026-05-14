import json
import os
import re
from datetime import datetime, timezone
from typing import Any


def start_experiment_run(
    experiments_dir: str,
    run_type: str,
    model_name: str,
    config: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, str]:
    """Create and persist a new experiment run context."""
    os.makedirs(experiments_dir, exist_ok=True)
    run_id = _build_run_id(run_type, model_name)
    run_dir = os.path.join(experiments_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    payload = {
        "run_id": run_id,
        "run_type": run_type,
        "model_name": model_name,
        "status": "running",
        "started_at": _utc_now(),
        "ended_at": None,
        "config": config,
        "dataset": dataset,
        "metrics": {},
        "quality_gates": {},
        "artifacts": [],
    }

    record_path = os.path.join(run_dir, "run.json")
    _write_json(record_path, payload)
    return {"run_id": run_id, "run_dir": run_dir, "record_path": record_path}


def finalize_experiment_run(
    run_context: dict[str, str],
    status: str,
    metrics: dict[str, Any],
    quality_gates: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Finalize an experiment run and persist outcomes."""
    record_path = run_context["record_path"]
    with open(record_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    payload["status"] = status
    payload["ended_at"] = _utc_now()
    payload["metrics"] = metrics
    payload["quality_gates"] = quality_gates or {}
    payload["artifacts"] = artifacts or []
    if notes:
        payload["notes"] = notes

    _write_json(record_path, payload)
    return payload


def _build_run_id(run_type: str, model_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    normalized_model = _slugify(model_name)
    normalized_type = _slugify(run_type)
    return f"{timestamp}_{normalized_model}_{normalized_type}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
