"""Common interface every compared model implements.

An adapter runs a model's own, unmodified code in that model's own Python
environment and returns results in the pipeline's shared formats:

- ``train()``  -> paths of the trained weights
- ``track()``  -> a folder of MOTChallenge files (one per validation sequence)
                 plus end-to-end FPS, ready for ``tir_jde.eval``
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tir_jde.experiment import run_dir

RUNNER = Path(__file__).with_name("_ultralytics_runner.py")


class Adapter:
    name = "base"

    def __init__(self, cfg: dict, manifest: dict):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.manifest = manifest
        self.data_root = Path(manifest["root"])
        self.run_dir = run_dir(cfg)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.python = str(self.model_cfg.get("python") or sys.executable)

    # ------------------------------------------------------------------ repo
    @property
    def repo_dir(self) -> Path:
        return Path(self.cfg["paths"]["baselines_dir"]) / self.model_cfg["name"]

    def ensure_repo(self) -> str:
        """Clones the baseline at the pinned commit (once) and returns the commit hash."""
        repo, commit = self.repo_dir, self.model_cfg.get("commit")
        if not (repo / ".git").exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            self._run(["git", "clone", "--quiet", self.model_cfg["repo"], str(repo)], "clone")
        if commit:
            current = self._git("rev-parse", "HEAD")
            if not current.startswith(commit):
                self._run(["git", "-C", str(repo), "fetch", "--quiet", "origin"], "fetch")
                self._run(["git", "-C", str(repo), "checkout", "--quiet", commit], "checkout")
        dirty = self._git("status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise RuntimeError(f"{repo} has local changes; baselines must run unmodified:\n{dirty}")
        return self._git("rev-parse", "HEAD")

    def _git(self, *args: str) -> str:
        out = subprocess.run(["git", "-C", str(self.repo_dir), *args], capture_output=True, text=True)
        return out.stdout.strip()

    # ------------------------------------------------------------------ processes
    def _run(self, cmd: list[str], log_name: str, cwd: Path | None = None, env: dict | None = None) -> float:
        """Runs a command, streaming output to the console and to logs/<log_name>.log."""
        log_file = self.logs_dir / f"{log_name}.log"
        full_env = {**os.environ, **(env or {})}
        t0 = time.perf_counter()
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(map(str, cmd))}\n")
            proc = subprocess.Popen(
                [str(c) for c in cmd], cwd=cwd, env=full_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Command failed ({code}), see {log_file}: {' '.join(map(str, cmd))}")
        return time.perf_counter() - t0

    def _run_job(self, job: dict, log_name: str) -> dict:
        """Runs _ultralytics_runner.py in the baseline's environment."""
        job_file = self.logs_dir / f"{log_name}.job.json"
        job["result_file"] = str(self.logs_dir / f"{log_name}.result.json")
        job_file.write_text(json.dumps(job, indent=2))
        self._run([self.python, str(RUNNER), str(job_file)], log_name)
        return json.loads(Path(job["result_file"]).read_text())

    # ------------------------------------------------------------------ interface
    def train(self) -> dict[str, str]:
        raise NotImplementedError

    def track(self, weights: dict[str, str]) -> dict:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def epoch_metrics(results_csv: Path) -> list[tuple[int, dict[str, float]]]:
        """Per-epoch metrics from an Ultralytics results.csv (for MLflow step metrics)."""
        if not results_csv.exists():
            return []
        rows = []
        with open(results_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items() if k}
                epoch = int(float(row.pop("epoch", len(rows))))
                metrics = {}
                for k, v in row.items():
                    try:
                        metrics[k.replace("(", "_").replace(")", "").replace("/", ".")] = float(v)
                    except (TypeError, ValueError):
                        pass
                rows.append((epoch, metrics))
        return rows
