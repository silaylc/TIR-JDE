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

Safety net (the tracking server must never cost a training run):
- every logging call is first appended to ``<run_dir>/mlflow_journal.jsonl``, then sent;
- if the server cannot be reached when a stage starts (3 attempts), the stage runs
  without it and the calls stay in the journal; a failed call during a stage only warns;
- after every stage the journal is copied into a second, local MLflow store
  (``<work_dir>/mlflow_backup.db``, artifacts in ``<work_dir>/mlartifacts_backup``);
  ``TIRJDE_MLFLOW_BACKUP=off`` disables it;
- ``python -m tir_jde.pipeline mlflow-sync <experiment>`` sends the calls the server missed.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Iterator

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


# --------------------------------------------------------------------------- journal
def _plain(o: Any) -> Any:
    """numpy scalars -> Python numbers (metrics must stay numbers); anything else -> str."""
    return o.item() if hasattr(o, "item") else str(o)


LOG_OPS = ("set_tag", "set_tags", "log_param", "log_params", "log_metric", "log_metrics",
           "log_dict", "log_artifact", "log_artifacts")
START_ATTEMPTS = int(os.environ.get("TIRJDE_MLFLOW_ATTEMPTS", "3"))
START_WAIT = float(os.environ.get("TIRJDE_MLFLOW_WAIT", "20"))


class Journal:
    """Append-only record of the MLflow calls of one run (``mlflow_journal.jsonl``) and the
    sync state of each store (``mlflow_sync.json``: run id, entries sent, first missed entry)."""

    _locks: ClassVar[dict[str, threading.Lock]] = {}

    def __init__(self, rdir: Path):
        self.file = rdir / "mlflow_journal.jsonl"
        self.state_file = rdir / "mlflow_sync.json"
        self.lock = Journal._locks.setdefault(str(self.file), threading.Lock())

    def append(self, op: str, args: list, kwargs: dict) -> int:
        line = json.dumps({"op": op, "args": args, "kwargs": kwargs, "time": time.time()}, default=_plain)
        with self.lock:
            n = self.count()
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return n

    def count(self) -> int:
        if not self.file.exists():
            return 0
        with open(self.file, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def entries(self, start: int = 0) -> list[dict]:
        if not self.file.exists():
            return []
        with open(self.file, encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for ln in lines[start:]:
            try:
                out.append(json.loads(ln))
            except ValueError:  # a line cut by a crash
                continue
        return out

    def state(self) -> dict:
        return json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def update_state(self, store: str, **values: Any) -> None:
        with self.lock:
            st = self.state()
            st.setdefault(store, {}).update(values)
            self.state_file.write_text(json.dumps(st, indent=2))

    def mark_missed(self, index: int) -> None:
        """The primary store missed entry ``index``; mlflow-sync resends from the first one."""
        st = self.state().get("primary", {})
        if st.get("missed_from") is None or index < st["missed_from"]:
            self.update_state("primary", missed_from=index)


class SafeMlflow:
    """Stands in for the ``mlflow`` module inside a stage: logging calls go to the journal
    first and then to the server (when there is one); a server error only warns."""

    def __init__(self, mlflow_module, journal: Journal):
        self._mlflow = mlflow_module
        self._journal = journal
        self._warned = False

    @property
    def online(self) -> bool:
        return self._mlflow is not None

    def __getattr__(self, name: str):
        if name in LOG_OPS:
            return lambda *a, **k: self._log(name, list(a), k)
        if self._mlflow is None:
            raise AttributeError(f"MLflow çevrimdışı: mlflow.{name} kullanılamaz")
        return getattr(self._mlflow, name)

    def active_run(self):
        if self._mlflow is None:
            return _OfflineRun()
        return self._mlflow.active_run()

    def _log(self, op: str, args: list, kwargs: dict) -> None:
        index = self._journal.append(op, args, kwargs)
        if self._mlflow is None:
            self._journal.mark_missed(index)
            return
        try:
            getattr(self._mlflow, op)(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - the server must never stop a run
            self._journal.mark_missed(index)
            if not self._warned:
                self._warned = True
                print(f"!!! [mlflow] sunucuya yazılamadı ({e}); kayıt journal'da, sonra: mlflow-sync", flush=True)


class _OfflineRun:
    class info:  # mirrors mlflow's Run.info
        run_id = None


def log_batch(rdir: Path, run_id: str | None, metrics: list[tuple[str, float, int, int]]) -> None:
    """Live metrics from background threads (key, value, timestamp ms, step): journal, then server."""
    journal = Journal(rdir)
    index = journal.append("log_batch", [[list(m) for m in metrics]], {})
    if run_id is None:
        journal.mark_missed(index)
        return
    try:
        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        MlflowClient().log_batch(run_id, metrics=[Metric(*m) for m in metrics])
    except Exception as e:  # noqa: BLE001
        journal.mark_missed(index)
        print(f"[mlflow] canlı kayıt sunucuya gitmedi (journal'da): {e}", flush=True)


def log_artifact_live(rdir: Path, run_id: str | None, path: str, artifact_path: str) -> None:
    """Artifact upload from a background thread: journal, then server."""
    journal = Journal(rdir)
    index = journal.append("log_artifact", [path], {"artifact_path": artifact_path})
    if run_id is None:
        journal.mark_missed(index)
        return
    try:
        from mlflow.tracking import MlflowClient

        MlflowClient().log_artifact(run_id, path, artifact_path=artifact_path)
    except Exception as e:  # noqa: BLE001
        journal.mark_missed(index)
        print(f"[mlflow] log sunucuya gitmedi (journal'da): {e}", flush=True)


def _start_primary(mlflow, cfg: dict, uri: str, run_id: str | None):
    """Opens the run on the tracking server; None if the server cannot be reached."""
    last = None
    for attempt in range(1, START_ATTEMPTS + 1):
        try:
            mlflow.set_tracking_uri(uri)
            _set_experiment(mlflow, cfg, uri)
            try:
                return mlflow.start_run(run_id=run_id, run_name=run_name(cfg))
            except Exception:  # run deleted or created on another tracking server
                if run_id is None:
                    raise
                return mlflow.start_run(run_name=run_name(cfg))
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < START_ATTEMPTS:
                print(f"[mlflow] sunucuya ulaşılamadı ({attempt}/{START_ATTEMPTS}): {e}", flush=True)
                time.sleep(START_WAIT)
    print(f"!!! [mlflow] {uri} kullanılamıyor ({last}). Aşama MLflow'suz sürüyor; kayıtlar journal'da "
          "ve yedek MLflow'da. Sonra: python -m tir_jde.pipeline mlflow-sync <deney> ...", flush=True)
    return None


@contextlib.contextmanager
def mlflow_run(cfg: dict, stage: str) -> Iterator["object"]:
    """Opens (or re-opens) the MLflow run of this experiment configuration.

    Yields a :class:`SafeMlflow`; whatever happens on the server, the calls are in the journal
    and the local backup store is refreshed when the stage ends."""
    import mlflow

    uri = tracking_uri(cfg)
    rdir = run_dir(cfg)
    journal = Journal(rdir)
    id_file = rdir / "mlflow_run_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else None
    first_stage = journal.count() == 0

    active = _start_primary(mlflow, cfg, uri, run_id)
    try:
        with (active or contextlib.nullcontext()):
            if active is not None:
                id_file.write_text(active.info.run_id)
                journal.update_state("primary", uri=uri, run_id=active.info.run_id)
            stored = journal.state().get("config_hash", {}).get("value")
            if active is not None:
                with contextlib.suppress(Exception):  # the server's tag wins when it can be read
                    stored = mlflow.get_run(active.info.run_id).data.tags.get("config.hash") or stored
            current = config_hash(cfg)
            if stored and stored != current:
                raise RuntimeError(
                    f"Run {run_name(cfg)} was created with a different configuration "
                    f"({stored} != {current}). Use a new seed or tag=... for changed settings."
                )
            journal.update_state("config_hash", value=current)
            safe = SafeMlflow(mlflow if active is not None else None, journal)
            safe.set_tags({"config.hash": current})
            safe.set_tags(
                {
                    "stage.last": stage,
                    "model": cfg.get("model", {}).get("name", ""),
                    "dataset": cfg.get("data", {}).get("name", ""),
                    "seed": str(cfg.get("seed", 0)),
                    "tir_jde.commit": git_commit(PROJ_ROOT),
                }
            )
            if first_stage:  # first stage of this run: log the full configuration once
                params = {k: (str(v)[:MAX_PARAM_LEN]) for k, v in flatten(cfg).items()}
                safe.log_params(params)
                conf = rdir / "config"
                conf.mkdir(exist_ok=True)
                save_config(cfg, conf / "config.yaml")
                (conf / "pipeline_env.txt").write_text(_pip_freeze(sys.executable))
                safe.log_artifacts(str(conf), artifact_path="config")
            yield safe
    finally:
        backup(cfg)


# --------------------------------------------------------------------------- replay
def backup_uri(cfg: dict) -> str | None:
    setting = os.environ.get("TIRJDE_MLFLOW_BACKUP", "")
    if setting.lower() in ("off", "0", "false", "no"):
        return None
    if setting:
        return setting
    db = (Path(cfg["paths"]["work_dir"]) / "mlflow_backup.db").resolve()
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db}"


def replay(cfg: dict, uri: str, entries: list[dict], run_id: str | None = None,
           artifacts_dir: str = "mlartifacts") -> str:
    """Writes journal entries into the run ``run_id`` (or a new run) of the store at ``uri``."""
    import mlflow

    previous = mlflow.get_tracking_uri()
    try:
        return _replay(mlflow, cfg, uri, entries, run_id, artifacts_dir)
    finally:
        mlflow.set_tracking_uri(previous)


def _replay(mlflow, cfg: dict, uri: str, entries: list[dict], run_id: str | None, artifacts_dir: str) -> str:
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    name = cfg.get("tracking", {}).get("experiment", "tir-jde")
    if uri.startswith("sqlite:") and mlflow.get_experiment_by_name(name) is None:
        artifacts = (Path(cfg["paths"]["work_dir"]) / artifacts_dir).resolve()
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_id=run_id, run_name=run_name(cfg)) as run:
        client = MlflowClient()
        for e in entries:
            op, args, kwargs = e["op"], e["args"], e["kwargs"]
            if op == "log_batch":
                client.log_batch(run.info.run_id, metrics=[Metric(*m) for m in args[0]])
            elif op in ("log_artifact", "log_artifacts") and not Path(args[0]).exists():
                print(f"[mlflow] artifact artık yok, atlandı: {args[0]}", flush=True)
            else:
                getattr(mlflow, op)(*args, **kwargs)
        return run.info.run_id


def backup(cfg: dict) -> None:
    """Copies the journal entries not yet in the local backup store (never raises)."""
    uri = backup_uri(cfg)
    if not uri:
        return
    rdir = run_dir(cfg)
    journal = Journal(rdir)
    st = journal.state().get("backup", {})
    if st.get("uri") not in (None, uri):
        st = {}
    sent = st.get("sent", 0)
    try:
        entries = journal.entries(sent)
        if not entries:
            return
        run_id = replay(cfg, uri, entries, st.get("run_id"), artifacts_dir="mlartifacts_backup")
        journal.update_state("backup", uri=uri, run_id=run_id, sent=sent + len(entries))
    except Exception as e:  # noqa: BLE001 - the backup must never stop a run
        print(f"[mlflow] yedek MLflow güncellenemedi: {e}", flush=True)


def sync_primary(cfg: dict) -> str | None:
    """Sends the journal entries the tracking server missed (from the first missed one on)."""
    rdir = run_dir(cfg)
    journal = Journal(rdir)
    st = journal.state().get("primary", {})
    start = st.get("missed_from")
    if start is None:
        print(f"[mlflow-sync] {run_name(cfg)}: eksik kayıt yok")
        return st.get("run_id")
    uri = tracking_uri(cfg)
    run_id = st.get("run_id") if st.get("uri") == uri else None
    entries = journal.entries(start if run_id else 0)
    run_id = replay(cfg, uri, entries, run_id)
    (rdir / "mlflow_run_id.txt").write_text(run_id)
    journal.update_state("primary", uri=uri, run_id=run_id, missed_from=None)
    print(f"[mlflow-sync] {run_name(cfg)}: {len(entries)} kayıt gönderildi -> {uri} ({run_id})")
    return run_id


def log_environment(mlflow, name: str, python: str, rdir: Path | None = None) -> None:
    """Records the package list of a baseline's own Python environment."""
    if rdir is None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / f"{name}_env.txt"
            f.write_text(_pip_freeze(python))
            mlflow.log_artifact(str(f), artifact_path="config")
        return
    f = rdir / "config" / f"{name}_env.txt"  # kept, so the journal can replay it
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(_pip_freeze(python))
    mlflow.log_artifact(str(f), artifact_path="config")
