"""Experiment pipeline: preprocess -> train -> evaluate -> visualize -> compare.

Usage (every command takes an experiment config plus optional key=value overrides)::

    python -m tir_jde.pipeline preprocess strong_baseline_bsb
    python -m tir_jde.pipeline train      strong_baseline_bsb seed=1
    python -m tir_jde.pipeline evaluate   strong_baseline_bsb seed=1
    python -m tir_jde.pipeline visualize  strong_baseline_bsb seed=1
    python -m tir_jde.pipeline run        dist_tracker_bsb seed=0 model.detector.epochs=2 tag=smoke
    python -m tir_jde.pipeline compare    strong_baseline_bsb dist_tracker_bsb --seeds 0 1 2
    python -m tir_jde.pipeline make-val   strong_baseline_bsb --fraction 0.15
    python -m tir_jde.pipeline eval-detector  common_detector_bsb tag=zeroshot
    python -m tir_jde.pipeline train-detector common_detector_bsb_pilot seed=0 tag=R0a
    python -m tir_jde.pipeline pilot-report   common_detector_bsb_pilot

Each experiment + seed (+ tag) is one MLflow run; every stage logs into it.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path

import numpy as np

from tir_jde import dataset
from tir_jde.experiment import load_config, resolve_path, run_dir, run_name


# --------------------------------------------------------------------------- stages
def stage_preprocess(cfg: dict) -> dict:
    manifest = dataset.prepare(cfg)
    print(json.dumps({k: manifest[k] for k in ("split_hash", "counts", "root")}, indent=2))
    return manifest


def _fastreid_metrics(path: Path) -> list[tuple[int, dict[str, float]]]:
    """FastReID's metrics.json (one JSON object per line) -> [(iteration, losses/lr)]."""
    import math

    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # line still being written
        step = row.pop("iteration", None)
        if step is None:
            continue
        keep = {k: float(v) for k, v in row.items()
                if (k.startswith("loss") or k in ("total_loss", "lr"))
                and isinstance(v, (int, float)) and math.isfinite(v)}
        if keep:
            rows.append((int(step), keep))
    return rows


class _LiveTrainLog:
    """While a model trains, sends finished epochs to MLflow and refreshes the
    training log there, so long runs can be followed on DagsHub."""

    def __init__(self, adapter, run_id: str, every: float = 60.0, log_every: float = 600.0):
        self.adapter, self.run_id = adapter, run_id
        self.every, self.log_every = every, log_every
        self.logged_epochs: set[int] = set()
        self.logged_reid_steps: set[int] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._last_log_upload = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
        self._push_epochs()

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            self._push_epochs()
            if time.monotonic() - self._last_log_upload >= self.log_every:
                self._last_log_upload = time.monotonic()
                self._push_logs()

    def _push_epochs(self) -> None:
        try:
            from mlflow.entities import Metric
            from mlflow.tracking import MlflowClient

            client, now = MlflowClient(), int(time.time() * 1000)
            # detector: one row per epoch in Ultralytics' results.csv
            for results in self.adapter.run_dir.glob("detector/**/results.csv"):
                for epoch, metrics in self.adapter.epoch_metrics(results):
                    if epoch in self.logged_epochs or not metrics:
                        continue
                    client.log_batch(self.run_id, metrics=[
                        Metric(f"train.{k}", v, now, epoch) for k, v in metrics.items()])
                    self.logged_epochs.add(epoch)
            # ReID (FastReID): one JSON line every 200 iterations and at each epoch end
            for step, metrics in _fastreid_metrics(self.adapter.run_dir / "reid" / "metrics.json"):
                if step in self.logged_reid_steps:
                    continue
                client.log_batch(self.run_id, metrics=[
                    Metric(f"reid.{k}", v, now, step) for k, v in metrics.items()])
                self.logged_reid_steps.add(step)
        except Exception as e:  # noqa: BLE001 - live logging must never stop training
            print(f"[mlflow] canlı kayıt atlandı: {e}", flush=True)

    def _push_logs(self) -> None:
        try:
            from mlflow.tracking import MlflowClient

            for log in self.adapter.logs_dir.glob("train_*.log"):
                MlflowClient().log_artifact(self.run_id, str(log), artifact_path="logs")
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] log yüklenemedi: {e}", flush=True)


class _GpuMonitor:
    """Samples GPU memory and utilisation with nvidia-smi while a model trains.

    Writes ``logs/gpu.csv``, sends the samples to MLflow once a minute and returns
    peak memory and mean utilisation at the end. Without nvidia-smi it does nothing."""

    QUERY = ("nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits")

    def __init__(self, logs_dir: Path, run_id: str | None, every: float = 15.0, push_every: float = 60.0):
        self.file = logs_dir / "gpu.csv"
        self.run_id, self.every, self.push_every = run_id, every, push_every
        self.samples: list[tuple[float, float, float, float]] = []  # seconds, used MiB, total MiB, util %
        self._pushed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._t0 = time.monotonic()

    def start(self) -> None:
        import shutil

        if shutil.which("nvidia-smi"):
            self.file.write_text("seconds,memory_used_mib,memory_total_mib,utilization_pct\n")
            self._thread.start()

    def _sample(self) -> None:
        import subprocess

        out = subprocess.run(self.QUERY, capture_output=True, text=True, timeout=20, check=False).stdout.strip()
        if not out:
            return
        used, total, util = (float(v) for v in out.splitlines()[0].split(","))
        row = (round(time.monotonic() - self._t0, 1), used, total, util)
        self.samples.append(row)
        with open(self.file, "a") as f:
            f.write(",".join(str(v) for v in row) + "\n")

    def _push(self) -> None:
        if not self.run_id or self._pushed >= len(self.samples):
            return
        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        now = int(time.time() * 1000)
        new = self.samples[self._pushed:]
        MlflowClient().log_batch(self.run_id, metrics=[
            m for sec, used, _, util in new
            for m in (Metric("gpu.memory_used_mib", used, now, int(sec)),
                      Metric("gpu.utilization_pct", util, now, int(sec)))])
        self._pushed = len(self.samples)

    def _loop(self) -> None:
        last_push = time.monotonic()
        while not self._stop.wait(self.every):
            try:
                self._sample()
                if time.monotonic() - last_push >= self.push_every:
                    last_push = time.monotonic()
                    self._push()
            except Exception as e:  # noqa: BLE001 - monitoring must never stop training
                print(f"[gpu] örnek atlandı: {e}", flush=True)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
        try:
            self._push()
        except Exception as e:  # noqa: BLE001
            print(f"[gpu] MLflow'a gönderilemedi: {e}", flush=True)
        if not self.samples:
            return {}
        used = [s[1] for s in self.samples]
        util = [s[3] for s in self.samples]
        return {"gpu.memory_peak_mib": max(used), "gpu.memory_total_mib": self.samples[0][2],
                "gpu.utilization_mean_pct": float(np.mean(util)),
                "gpu.utilization_median_pct": float(np.median(util))}


def _log_detector_extras(mlflow, weights: dict, run_dir_: Path) -> None:
    """Frame selection (steps, counts, hashes, lists) and the full-validation metrics."""
    frames_file = Path(weights.get("detector_frames", ""))
    if frames_file.is_file():
        frames = json.loads(frames_file.read_text())
        mlflow.set_tags({f"data.{part}_frames_hash": v["hash"] for part, v in frames.items()})
        mlflow.log_metrics({f"data.{part}_{k}": float(v[k]) for part, v in frames.items() for k in ("step", "frames")})
        mlflow.log_artifact(str(frames_file), artifact_path="data")
        lists = run_dir_ / "detector_data"
        if lists.is_dir():
            mlflow.log_artifacts(str(lists), artifact_path="data/frame_lists")
        per_epoch = frames["train"].get("iterations_per_epoch")
        results = Path(weights.get("detector_results", ""))
        if per_epoch and results.is_file():
            from tir_jde.modeling.adapters.base import Adapter

            secs = [m["epoch_seconds"] for _, m in Adapter.epoch_metrics(results) if "epoch_seconds" in m]
            if secs:
                mlflow.log_metrics({"train.iterations_per_epoch": float(per_epoch),
                                    "train.seconds_per_iteration": float(np.mean(secs)) / per_epoch})
    val_file = Path(weights.get("detector_val_full", ""))
    if val_file.is_file():
        val = json.loads(val_file.read_text())
        mlflow.log_metrics({f"det_val_full.{_metric_name(k)}": v for k, v in val["metrics"].items()})


def _metric_name(key: str) -> str:
    return key.replace("(", "_").replace(")", "").replace("/", ".")


def stage_train(cfg: dict, detector_only: bool = False) -> dict:
    """Trains the model (detector + ReID where the model has one); ``detector_only``
    stops after the detector (pilot runs)."""
    from tir_jde.modeling.adapters import get_adapter
    from tir_jde.tracking import log_environment, mlflow_run

    manifest = dataset.prepare(cfg)
    adapter = get_adapter(cfg, manifest)
    commit = adapter.ensure_repo()
    with mlflow_run(cfg, "train") as mlflow:
        mlflow.set_tags({"baseline.commit": commit, "data.split_hash": manifest["split_hash"]})
        mlflow.log_params({f"data.count.{k}": v for k, v in manifest["counts"].items()})
        mlflow.log_dict(manifest["sequences"], "data/sequences.json")
        log_environment(mlflow, cfg["model"]["name"], adapter.python)

        run_id = mlflow.active_run().info.run_id
        live = _LiveTrainLog(adapter, run_id)
        gpu = _GpuMonitor(adapter.logs_dir, run_id)
        live.start()
        gpu.start()
        try:
            weights = adapter.train_detector() if detector_only else adapter.train()
        finally:
            live.stop()
            gpu_summary = gpu.stop()
        if gpu_summary:
            mlflow.log_metrics(gpu_summary)
            (adapter.run_dir / "gpu_summary.json").write_text(json.dumps(gpu_summary, indent=2))
            mlflow.log_artifact(str(gpu.file), artifact_path="logs")
        (adapter.run_dir / "weights.json").write_text(json.dumps(weights, indent=2))
        _log_detector_extras(mlflow, weights, adapter.run_dir)

        for epoch, metrics in adapter.epoch_metrics(Path(weights.get("detector_results", ""))):
            if epoch not in live.logged_epochs:
                mlflow.log_metrics({f"train.{k}": v for k, v in metrics.items()}, step=epoch)
        if Path(weights["detector"]).exists():
            mlflow.log_artifact(weights["detector"], artifact_path="weights")
        for log in adapter.logs_dir.glob("train_*.log"):
            mlflow.log_artifact(str(log), artifact_path="logs")
    return weights


def stage_evaluate(cfg: dict) -> dict:
    from tir_jde.eval.protocol import first_frame_check
    from tir_jde.eval.trackeval_runner import evaluate
    from tir_jde.modeling.adapters import get_adapter
    from tir_jde.tracking import mlflow_run

    manifest = dataset.prepare(cfg)
    adapter = get_adapter(cfg, manifest)
    adapter.ensure_repo()
    weights_file = adapter.run_dir / "weights.json"
    if not weights_file.exists():
        raise FileNotFoundError(f"No trained weights for {run_name(cfg)}; run 'train' first.")
    weights = json.loads(weights_file.read_text())

    with mlflow_run(cfg, "evaluate") as mlflow:
        tracked = adapter.track(weights)
        gt_dir = Path(manifest["root"]) / "gt" / "val"
        result = evaluate(gt_dir, tracked["pred_dir"], manifest["sequences"]["val"],
                          metrics=tuple(cfg.get("eval", {}).get("metrics", ("HOTA", "CLEAR", "Identity"))))
        result["fps"] = tracked["fps"]
        result["frames"] = tracked["frames"]
        # BSB protocol: frame 1 of the output must reproduce the given first-frame boxes
        result["protocol"] = first_frame_check(gt_dir, Path(tracked["pred_dir"]), manifest["sequences"]["val"])
        mlflow.log_metrics({"protocol.first_frame_match": result["protocol"]["match"],
                            "protocol.first_frame_extra": result["protocol"]["extra"]})
        if result["protocol"]["match"] < 1.0 or result["protocol"]["extra"]:
            print(f"!!! PROTOKOL UYARISI: 1. kare verilen kutularla uyuşmuyor {result['protocol']}", flush=True)

        mlflow.log_metrics({f"val.{k}": float(v) for k, v in result["overall"].items()})
        mlflow.log_metric("val.FPS_end_to_end", float(tracked["fps"]))
        mlflow.log_dict(result, "eval/results.json")
        mlflow.log_artifacts(tracked["pred_dir"], artifact_path="eval/predictions")
        (adapter.run_dir / "results.json").write_text(json.dumps(result, indent=2))
        _write_per_sequence_csv(result["per_sequence"], adapter.run_dir / "per_sequence.csv")
        mlflow.log_artifact(str(adapter.run_dir / "per_sequence.csv"), artifact_path="eval")
        for log in adapter.logs_dir.glob("track*.log"):
            mlflow.log_artifact(str(log), artifact_path="logs")
    print(json.dumps(result["overall"], indent=2), f"\nFPS (end-to-end): {tracked['fps']:.1f}",
          f"\nProtokol (1. kare = verilen kutular): {result['protocol']['match']:.3f}, fazla kutu: "
          f"{result['protocol']['extra']}")
    return result


def stage_visualize(cfg: dict, sequences: list[str] | None = None, max_frames: int = 300) -> list[Path]:
    from tir_jde import plots
    from tir_jde.tracking import mlflow_run

    manifest = dataset.prepare(cfg)
    rdir = run_dir(cfg)
    results_file = rdir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No evaluation results for {run_name(cfg)}; run 'evaluate' first.")
    result = json.loads(results_file.read_text())
    seqs = sequences or manifest["sequences"]["val"][:2]
    root = Path(manifest["root"])
    outputs = [plots.per_sequence_chart(result["per_sequence"], "HOTA", rdir / "figures" / "hota_per_sequence.png",
                                        title=f"{run_name(cfg)} — HOTA per sequence")]
    for seq in seqs:
        outputs.append(plots.draw_tracks(
            root / "track" / "val" / "Test_imgs" / seq, rdir / "predictions" / f"{seq}.txt",
            rdir / "figures" / f"{seq}.mp4", gt_file=root / "gt" / "val" / f"{seq}.txt", max_frames=max_frames,
        ))
    with mlflow_run(cfg, "visualize") as mlflow:
        for out in outputs:
            mlflow.log_artifact(str(out), artifact_path="figures")
    return outputs


def stage_compare(experiments: list[str], seeds: list[int], overrides: list[str], out_dir: Path) -> Path:
    """Mean ± std over seeds for each experiment, as CSV and a bar chart."""
    from tir_jde import plots

    metrics = ["HOTA", "DetA", "AssA", "IDF1", "MOTA"]
    table: dict[str, dict[str, tuple[float, float]]] = {}
    rows = []
    for exp in experiments:
        values: dict[str, list[float]] = {m: [] for m in metrics + ["IDSW", "FPS"]}
        for seed in seeds:
            cfg = load_config(exp, [*overrides, f"seed={seed}"])
            f = run_dir(cfg) / "results.json"
            if not f.exists():
                print(f"[compare] missing: {run_name(cfg)}")
                continue
            r = json.loads(f.read_text())
            for m in metrics + ["IDSW"]:
                values[m].append(r["overall"].get(m, np.nan))
            values["FPS"].append(r.get("fps", np.nan))
        name = load_config(exp, overrides)["name"]
        table[name] = {}
        for m, v in values.items():
            finite = [x for x in v if x is not None and np.isfinite(x)]
            if finite:
                table[name][m] = (float(np.mean(finite)), float(np.std(finite)))
        rows.append({"experiment": name, "seeds": len(values["HOTA"]),
                     **{m: f"{table[name][m][0]:.4f} ± {table[name][m][1]:.4f}" for m in table[name]}})
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "comparison.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["experiment"])
        writer.writeheader()
        writer.writerows(rows)
    plots.comparison_chart(table, metrics, out_dir / "comparison.png")
    print(csv_file.read_text())
    return csv_file


def stage_eval_detector(cfg: dict) -> dict:
    """Zero-shot check: the starting weights (``detector.init``) on every val frame, no training."""
    from tir_jde.modeling.adapters import get_adapter
    from tir_jde.tracking import mlflow_run

    manifest = dataset.prepare(cfg)
    adapter = get_adapter(cfg, manifest)
    commit = adapter.ensure_repo()
    with mlflow_run(cfg, "eval_detector") as mlflow:
        mlflow.set_tags({"baseline.commit": commit, "data.split_hash": manifest["split_hash"]})
        result = adapter.evaluate_init_weights()
        mlflow.set_tag("detector.init_weights", Path(result["weights"]).name)
        mlflow.log_metrics({f"det_val_zero_shot.{_metric_name(k)}": v for k, v in result["metrics"].items()})
    print(json.dumps(result["metrics"], indent=2))
    return result


def stage_pilot_report(cfg: dict, baseline: str = "R0a", noise_pair: tuple[str, str] = ("R0a", "R0b"),
                       metrics: tuple[str, ...] = ("mAP50", "recall")) -> Path:
    """Table of the detector pilot runs and the decision rule.

    Noise floor = |a - b| for the two runs of noise_pair (same settings, another seed).
    A run's setting is adopted when it beats baseline (the defaults with the chosen
    starting weights) by more than twice the noise floor on the first metric (mAP50);
    recall is reported next to it. Two seeds give only a rough noise estimate, which the
    report states."""
    base = Path(cfg["paths"]["work_dir"]) / "runs"
    prefix = f"{cfg['name']}-s"
    runs: dict[str, dict] = {}
    for d in sorted(base.glob(f"{prefix}*-*")):
        seed_tag = d.name[len(prefix):]
        seed, _, tag = seed_tag.partition("-")
        val_file = d / "detector_val_full.json"
        if not tag or not val_file.exists():
            continue
        val = json.loads(val_file.read_text())["metrics"]
        row = {"seed": int(seed),
               **{m: next((v for k, v in val.items() if k.endswith(f"/{m}(B)")), np.nan) for m in metrics}}
        res = d / "detector" / "train" / "results.csv"
        if res.exists():
            from tir_jde.modeling.adapters.base import Adapter
            secs = [m.get("epoch_seconds") for _, m in Adapter.epoch_metrics(res) if "epoch_seconds" in m]
            row["epoch_min"] = float(np.mean(secs)) / 60 if secs else np.nan
            fr = d / "detector_frames.json"
            per_epoch = json.loads(fr.read_text())["train"].get("iterations_per_epoch") if fr.exists() else None
            row["s_per_it"] = float(np.mean(secs)) / per_epoch if secs and per_epoch else np.nan
        gpu_file = d / "gpu_summary.json"
        if gpu_file.exists():
            g = json.loads(gpu_file.read_text())
            row["gpu_peak_gib"] = g.get("gpu.memory_peak_mib", np.nan) / 1024
            row["gpu_util_pct"] = g.get("gpu.utilization_mean_pct", np.nan)
        runs[tag] = row
    if not runs:
        print(f"[pilot-report] {base} altında tamamlanmış pilot koşusu yok ({prefix}<tohum>-<etiket>)")
        return base

    lines = [f"# Dedektör pilotu: {cfg['name']}", ""]
    first = metrics[0]
    a, b = noise_pair
    noise = abs(runs[a][first] - runs[b][first]) if {a, b} <= set(runs) else np.nan
    cols = ["seed", *metrics, f"Δ{first}", "karar", "epoch_min", "s_per_it", "gpu_peak_gib", "gpu_util_pct"]
    lines.append("| koşu | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for tag, r in runs.items():
        delta = r[first] - runs[baseline][first] if baseline in runs else np.nan
        if tag == baseline:
            decision = "referans"
        elif tag in noise_pair:
            decision = "gürültü"
        elif np.isfinite(noise) and np.isfinite(delta):
            decision = "benimse" if delta > 2 * noise else "varsayılan kalır"
        else:
            decision = "?"
        cells = [r["seed"], *[r.get(m, np.nan) for m in metrics], delta, decision,
                 r.get("epoch_min", np.nan), r.get("s_per_it", np.nan), r.get("gpu_peak_gib", np.nan),
                 r.get("gpu_util_pct", np.nan)]
        lines.append(f"| {tag} | " + " | ".join(f"{c:.4f}" if isinstance(c, float) else str(c) for c in cells) + " |")
    lines += ["", (f"Referans: {baseline}. Gürültü tabanı |{a} − {b}| ({first}) = {noise:.4f}; "
                   f"eşik 2× = {2 * noise:.4f}."),
              "Gürültü yalnızca iki tohumla tahmin edildi; eşik kaba bir eşiktir."]
    out = base / f"{cfg['name']}-pilot_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    return out


def stage_make_val(cfg: dict, per_family: int, seed: int, force: bool,
                   fraction: float | None = None) -> Path:
    data_cfg = cfg["data"]
    out = resolve_path(data_cfg["splits"]["val"])
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; split lists are fixed once created (use --force to overwrite).")
    excluded = set(data_cfg.get("exclude") or [])
    train = [s for s in dataset.read_split(data_cfg["splits"]["train"]) if s not in excluded]
    families = {s: f for s in train if (f := dataset.family_of(s))}
    groups = dataset.read_scene_groups(data_cfg.get("scene_groups"))
    val = dataset.make_val_split(train, families, per_family, seed, fraction, groups)
    out.parent.mkdir(parents=True, exist_ok=True)
    rule = f"{fraction:g} of each family (min 1)" if fraction is not None else f"{per_family} per family"
    if groups:
        rule += ", scene groups kept whole"
    if excluded:
        rule += f", excluded: {' '.join(sorted(excluded))}"
    out.write_text(f"# validation group: {rule}, seed {seed}\n" + "\n".join(val) + "\n")
    print(f"{len(val)} validation sequences -> {out}")
    return out


def _write_per_sequence_csv(per_sequence: dict, path: Path) -> None:
    keys = sorted({k for v in per_sequence.values() for k in v})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", *keys])
        for seq, values in sorted(per_sequence.items()):
            writer.writerow([seq, *[values.get(k, "") for k in keys]])


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tir_jde.pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preprocess", "train", "train-detector", "eval-detector", "evaluate", "visualize", "run",
                 "pilot-report"):
        p = sub.add_parser(name)
        p.add_argument("experiment")
        p.add_argument("overrides", nargs="*")
        if name == "pilot-report":
            p.add_argument("--baseline", default="R0a", help="reference run (defaults, chosen weights)")
            p.add_argument("--noise", nargs=2, default=["R0a", "R0b"], help="two seeds of the same settings")
        if name in ("visualize", "run"):
            p.add_argument("--sequences", nargs="*")
    p = sub.add_parser("compare")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--set", dest="overrides", nargs="*", default=[])
    p.add_argument("--out", default="reports/comparison")
    p = sub.add_parser("make-val")
    p.add_argument("experiment")
    p.add_argument("--per-family", type=int, default=1)
    p.add_argument("--fraction", type=float, default=None,
                   help="share of each family for validation (min 1); overrides --per-family")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "compare":
        stage_compare(args.experiments, args.seeds, args.overrides, resolve_path(args.out))
        return
    cfg = load_config(args.experiment, getattr(args, "overrides", []))
    if args.command == "make-val":
        stage_make_val(cfg, args.per_family, args.seed, args.force, args.fraction)
    elif args.command == "preprocess":
        stage_preprocess(cfg)
    elif args.command == "train":
        stage_train(cfg)
    elif args.command == "train-detector":
        stage_train(cfg, detector_only=True)
    elif args.command == "eval-detector":
        stage_eval_detector(cfg)
    elif args.command == "pilot-report":
        stage_pilot_report(cfg, args.baseline, tuple(args.noise))
    elif args.command == "evaluate":
        stage_evaluate(cfg)
    elif args.command == "visualize":
        stage_visualize(cfg, args.sequences)
    elif args.command == "run":
        stage_train(cfg)
        stage_evaluate(cfg)
        stage_visualize(cfg, args.sequences)


if __name__ == "__main__":
    main()
