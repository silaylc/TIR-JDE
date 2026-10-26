"""Common interface every compared model implements.

An adapter runs a model's own, unmodified code in that model's own Python
environment and returns results in the pipeline's shared formats:

- ``train()``  -> paths of the trained weights
- ``track()``  -> a folder of MOTChallenge files (one per validation sequence)
                 plus end-to-end FPS, ready for ``tir_jde.eval``
"""

from __future__ import annotations

import collections
import csv
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tir_jde.experiment import run_dir

RUNNER = Path(__file__).with_name("_ultralytics_runner.py")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

# Console output is throttled so long trainings do not flood (and crash) the
# notebook page; the full output always goes to logs/<name>.log.
PROGRESS_RE = re.compile(r"\d+%\|.*\||\d+/\d+ \[\d")       # tqdm-style progress bars
PROGRESS_SECONDS = float(os.environ.get("TIRJDE_PROGRESS_SECONDS", "60"))
MAX_LINES_PER_MINUTE = int(os.environ.get("TIRJDE_MAX_LINES_PER_MINUTE", "40"))


class _ConsoleThrottle:
    """Progress bars: at most one line every PROGRESS_SECONDS. Other lines: up to
    MAX_LINES_PER_MINUTE, the rest are counted and reported."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.last_progress = 0.0
        self.window_start = time.monotonic()
        self.window_lines = 0
        self.skipped = 0
        self.tail: collections.deque[str] = collections.deque(maxlen=15)  # last hidden lines

    def write(self, line: str) -> None:
        now = time.monotonic()
        if PROGRESS_RE.search(line):
            if now - self.last_progress >= PROGRESS_SECONDS:
                self.last_progress = now
                sys.stdout.write(line.rstrip()[-180:] + "\n")
                sys.stdout.flush()
            return
        if now - self.window_start >= 60:
            self._report()
            self.window_start, self.window_lines = now, 0
        if self.window_lines < MAX_LINES_PER_MINUTE:
            self.window_lines += 1
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            self.skipped += 1
            self.tail.append(line)

    def _report(self) -> None:
        if self.skipped:
            sys.stdout.write(f"  ... {self.skipped} satır ekrana yazılmadı (tam log: {self.log_file})\n")
            self.skipped = 0
            self.tail.clear()

    def close(self) -> None:
        """Reports hidden lines and shows the last ones (errors usually end up there)."""
        last = list(self.tail)
        self._report()
        if last:
            sys.stdout.write("  --- ekrana yazılmayan son satırlar ---\n" + "".join(last))
            sys.stdout.flush()


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
        # Files the baseline's own code regenerates at run time (e.g. FastReID re-cythonizes its
        # evaluation module) are restored instead of being treated as code changes.
        generated = self.model_cfg.get("generated_files") or []
        changed = self._git("diff", "--name-only", "HEAD").splitlines()
        restore = [f for f in changed if any(fnmatch.fnmatch(f, pat) for pat in generated)]
        if restore:
            self._run(["git", "-C", str(repo), "checkout", "--", *restore], "restore_generated")
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
        full_env = {**os.environ, "MPLBACKEND": "Agg", **(env or {})}  # headless plots
        # The baseline's own environment first on PATH, so tools its code starts by
        # name (e.g. FastReID's ``make`` -> ``python3 setup.py``) use that environment.
        env_bin = Path(str(self.python)).parent
        if env_bin != Path("."):
            full_env["PATH"] = f"{env_bin}{os.pathsep}{full_env.get('PATH', '')}"
        t0 = time.perf_counter()
        console = _ConsoleThrottle(log_file)
        print(f"[{log_name}] tam log: {log_file}", flush=True)
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(map(str, cmd))}\n")
            proc = subprocess.Popen(
                [str(c) for c in cmd], cwd=cwd, env=full_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                console.write(line)
            code = proc.wait()
            console.close()
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

    # ------------------------------------------------------------------ detector
    def frame_steps(self) -> dict[str, int]:
        det = self.model_cfg["detector"]
        return {"train": int(det.get("frame_step") or 1), "val": int(det.get("val_frame_step") or 1)}

    def detector_data_yaml(self) -> Path:
        """Dataset file the detector trains on: the full one from preprocessing, or the
        run's own frame lists when ``frame_step`` / ``val_frame_step`` thin out frames."""
        if self.frame_steps() == {"train": 1, "val": 1}:
            return self.data_root / "yolo" / "dataset.yaml"
        return self.run_dir / "detector_data" / "dataset.yaml"

    def write_detector_data(self) -> dict:
        """Writes the frame lists (absolute image paths) and returns steps, counts and hashes.

        The lists stay in the run folder and go to MLflow, so the exact frames are known."""
        from tir_jde.dataset import frames_hash, select_frames

        yolo = self.data_root / "yolo"
        steps = self.frame_steps()
        info: dict = {}
        out = self.detector_data_yaml().parent
        for part, step in steps.items():
            images = [p for p in (yolo / part / "images").iterdir() if not p.name.startswith(".")]
            chosen = select_frames(images, step)
            info[part] = {"step": step, "frames": len(chosen), "of": len(images), "hash": frames_hash(chosen)}
            if out != yolo:
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{part}.txt").write_text("".join(f"{p}\n" for p in chosen))
        det = self.model_cfg["detector"]
        fraction = float((det.get("extra") or {}).get("fraction", 1.0))
        used = round(info["train"]["frames"] * fraction)
        info["train"]["iterations_per_epoch"] = math.ceil(used / det["batch"])  # batch başına bir iterasyon
        if out != yolo:
            (out / "dataset.yaml").write_text(
                f"path: {out}\ntrain: {out / 'train.txt'}\nval: {out / 'val.txt'}\nnc: 1\nnames: ['uav']\n")
        (self.run_dir / "detector_frames.json").write_text(json.dumps(info, indent=2))
        return info

    def init_weights(self) -> str | None:
        """Starting weights of the detector (``detector.init``).

        ``scratch`` -> None (architecture yaml); a key of ``detector.pretrained`` (e.g. ``coco``,
        ``sot``) -> that checkpoint, downloaded once and checked against its SHA-256 when one is
        given; anything else -> that file path."""
        det = self.model_cfg["detector"]
        init = det.get("init", "scratch")
        if init in (None, "scratch"):
            return None
        entry = (det.get("pretrained") or {}).get(init)
        if entry is None:
            return str(init)
        arch = det.get("arch")
        if entry.get("arch") and arch and entry["arch"] != arch:
            raise ValueError(f"'{init}' ağırlığı {entry['arch']} mimarisi için; config'te arch={arch}")
        url = entry["url"].format(arch=arch)
        dest = Path(self.cfg["paths"]["baselines_dir"]) / "weights" / f"{self.name}_{init}_{Path(url).name}"
        if not dest.exists():
            import urllib.request

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            print(f"[init] '{init}' ağırlığı indiriliyor: {url}", flush=True)
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
        expected = entry.get("sha256")
        if expected:
            actual = _sha256(dest)
            if actual != expected:
                raise RuntimeError(f"{dest} SHA-256 uyuşmuyor: {actual} != {expected} (dosyayı silip yeniden indirin)")
        return str(dest)

    def workers(self, default: int = 8) -> int:
        """``detector.workers``: a number, or ``auto`` = the machine's CPU count."""
        w = self.model_cfg["detector"].get("workers", default)
        if w in (None, "auto"):
            return os.cpu_count() or default
        return int(w)

    def detector_val_job(self, weights: str) -> dict:
        """Full-validation job (every frame) for a trained detector."""
        det = self.model_cfg["detector"]
        extra = det.get("extra") or {}
        return {
            "mode": "val",
            "lib_root": self.detector_job()["lib_root"],
            "cwd": self.detector_job()["cwd"],
            "weights": weights,
            "val_args": {
                "data": str(self.data_root / "yolo" / "dataset.yaml"),
                "split": "val",
                "imgsz": det["imgsz"],
                "batch": det["batch"],
                "conf": extra.get("conf", 0.001),
                "single_cls": True,
                "device": 0,
                "workers": self.workers(),
                "project": str(self.run_dir / "detector"),
                "name": "val_full",
                "exist_ok": True,
            },
        }

    def detector_job(self) -> dict:
        raise NotImplementedError

    def train_detector(self) -> dict[str, str]:
        """Trains the detector; with thinned validation frames, validates on every frame afterwards."""
        frames = self.write_detector_data()
        result = self._run_job(self.detector_job(), "train_detector")
        weights = {"detector": result["best"], "detector_results": str(Path(result["save_dir"]) / "results.csv"),
                   "detector_frames": str(self.run_dir / "detector_frames.json")}
        if frames["val"]["step"] > 1:
            val = self._run_job(self.detector_val_job(result["best"]), "val_detector")
            out = self.run_dir / "detector_val_full.json"
            out.write_text(json.dumps(val, indent=2))
            weights["detector_val_full"] = str(out)
        return weights

    def evaluate_init_weights(self) -> dict:
        """Validates the starting weights on every val frame without training (zero-shot)."""
        weights = self.init_weights()
        if weights is None:
            raise ValueError("detector.init=scratch: değerlendirilecek başlangıç ağırlığı yok")
        result = self._run_job(self.detector_val_job(weights), "val_zero_shot")
        result["weights"] = weights
        (self.run_dir / "detector_val_zero_shot.json").write_text(json.dumps(result, indent=2))
        return result

    # ------------------------------------------------------------------ interface
    def train(self) -> dict[str, str]:
        return self.train_detector()

    def track(self, weights: dict[str, str]) -> dict:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def epoch_metrics(results_csv: Path) -> list[tuple[int, dict[str, float]]]:
        """Per-epoch metrics from an Ultralytics results.csv (for MLflow step metrics)."""
        if not results_csv.exists():
            return []
        rows = []
        previous_time = 0.0
        with open(results_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items() if k}
                epoch = int(float(row.pop("epoch", len(rows))))
                metrics = {}
                for k, v in row.items():
                    try:
                        value = float(v)
                    except (TypeError, ValueError):
                        continue
                    name = k.replace("(", "_").replace(")", "").replace("/", ".")
                    if math.isfinite(value):
                        metrics[name] = value
                    else:  # nan/inf (e.g. Dist-Tracker's box loss): MLflow gets a 1.0 flag instead
                        metrics[f"{name}_nan"] = 1.0
                if "time" in metrics:  # Ultralytics writes the cumulative training time
                    metrics["epoch_seconds"] = metrics["time"] - previous_time
                    previous_time = metrics["time"]
                rows.append((epoch, metrics))
        return rows
