"""MLflow logging shared by every pipeline stage.

One experiment configuration maps to one MLflow run. Each stage (train,
evaluate, visualize) re-opens the same run, so parameters, metrics and
artifacts of a model end up in a single place. The run id is stored in
``<run_dir>/mlflow_run_id.txt``.

Tracking server, in order of precedence:
1. ``MLFLOW_TRACKING_URI`` environment variable (plus the usual MLflow
   credentials variables);
2. DagsHub, when ``DAGSHUB_USER`` and ``DAGSHUB_TOKEN`` are set — the run goes
   to ``https://dagshub.com/<owner>/<repo>.mlflow`` from ``tracking.dagshub``;
3. a local file store at ``<work_dir>/mlruns``.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from tir_jde.experiment import PROJ_ROOT, config_hash, flatten, run_dir, run_name, save_config

MAX_PARAM_LEN = 500  # MLflow rejects longer parameter values


def tracking_uri(cfg: dict) -> str:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    dagshub = cfg.get("tracking", {}).get("dagshub") or {}
    if os.environ.get("DAGSHUB_USER") and os.environ.get("DAGSHUB_TOKEN") and dagshub.get("repo"):
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.environ["DAGSHUB_USER"])
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ["DAGSHUB_TOKEN"])
        owner = dagshub.get("owner") or os.environ["DAGSHUB_USER"]
        return f"https://dagshub.com/{owner}/{dagshub['repo']}.mlflow"
    return (Path(cfg["paths"]["work_dir"]) / "mlruns").resolve().as_uri()


def git_commit(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pip_freeze(python: str) -> str:
    try:
        out = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        return out.stdout
    except Exception as e:  # noqa: BLE001 - logging must never break a run
        return f"pip freeze failed: {e}"


@contextlib.contextmanager
def mlflow_run(cfg: dict, stage: str) -> Iterator["object"]:
    """Opens (or re-opens) the MLflow run of this experiment configuration."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(cfg))
    mlflow.set_experiment(cfg.get("tracking", {}).get("experiment", "tir-jde"))

    rdir = run_dir(cfg)
    id_file = rdir / "mlflow_run_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else None

    try:
        active = mlflow.start_run(run_id=run_id, run_name=run_name(cfg))
    except Exception:  # run deleted or created on another tracking server
        active = mlflow.start_run(run_name=run_name(cfg))
    id_file.write_text(active.info.run_id)

    with active:
        stored = mlflow.get_run(active.info.run_id).data.tags.get("config.hash")
        current = config_hash(cfg)
        if stored and stored != current:
            raise RuntimeError(
                f"Run {run_name(cfg)} was created with a different configuration "
                f"({stored} != {current}). Use a new seed or tag=... for changed settings."
            )
        mlflow.set_tags(
            {"config.hash": current}
        )
        mlflow.set_tags(
            {
                "stage.last": stage,
                "model": cfg.get("model", {}).get("name", ""),
                "dataset": cfg.get("data", {}).get("name", ""),
                "seed": str(cfg.get("seed", 0)),
                "tir_jde.commit": git_commit(PROJ_ROOT),
            }
        )
        if run_id is None:  # first stage of this run: log the full configuration once
            params = {k: (str(v)[:MAX_PARAM_LEN]) for k, v in flatten(cfg).items()}
            mlflow.log_params(params)
            with tempfile.TemporaryDirectory() as tmp:
                save_config(cfg, Path(tmp) / "config.yaml")
                Path(tmp, "pipeline_env.txt").write_text(_pip_freeze(sys.executable))
                mlflow.log_artifacts(tmp, artifact_path="config")
        yield mlflow


def log_environment(mlflow, name: str, python: str) -> None:
    """Records the package list of a baseline's own Python environment."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / f"{name}_env.txt"
        f.write_text(_pip_freeze(python))
        mlflow.log_artifact(str(f), artifact_path="config")
