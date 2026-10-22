"""MLflow logging shared by every pipeline stage.

One experiment configuration maps to one MLflow run. Each stage (train,
evaluate, visualize) re-opens the same run, so parameters, metrics and
artifacts of a model end up in a single place. The run id is stored in
``<run_dir>/mlflow_run_id.txt``.

Tracking server, in order of precedence:
1. ``MLFLOW_TRACKING_URI`` environment variable (plus the usual MLflow
   credentials variables);
2. DagsHub, when ``DAGSHUB_TOKEN`` is set — the run goes to
   ``https://dagshub.com/<owner>/<repo>.mlflow`` from ``tracking.dagshub``;
3. a local SQLite store at ``<work_dir>/mlflow.db`` with artifacts in
   ``<work_dir>/mlartifacts`` (MLflow no longer accepts new file stores).
   View it with ``mlflow ui --backend-store-uri sqlite:///<work_dir>/mlflow.db``.
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
    token = os.environ.get("DAGSHUB_TOKEN")
    owner = dagshub.get("owner") or os.environ.get("DAGSHUB_USER")
    if token and owner and dagshub.get("repo"):
        # DagsHub accepts the token as both user name and password, so a wrong or
        # missing DAGSHUB_USER cannot break the login. Always overrides older
        # MLFLOW_* values left in the environment.
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        return f"https://dagshub.com/{owner}/{dagshub['repo']}.mlflow"
    db = (Path(cfg["paths"]["work_dir"]) / "mlflow.db").resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db}"


def _set_experiment(mlflow, cfg: dict, uri: str) -> None:
    name = cfg.get("tracking", {}).get("experiment", "tir-jde")
    if uri.startswith("sqlite:") and mlflow.get_experiment_by_name(name) is None:
        artifacts = (Path(cfg["paths"]["work_dir"]) / "mlartifacts").resolve()
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)


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

    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    _set_experiment(mlflow, cfg, uri)

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
        mlflow.set_tags({"config.hash": current})
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
