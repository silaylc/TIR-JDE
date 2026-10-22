"""Experiment pipeline: preprocess -> train -> evaluate -> visualize -> compare.

Usage (every command takes an experiment config plus optional key=value overrides)::

    python -m tir_jde.pipeline preprocess strong_baseline_bsb
    python -m tir_jde.pipeline train      strong_baseline_bsb seed=1
    python -m tir_jde.pipeline evaluate   strong_baseline_bsb seed=1
    python -m tir_jde.pipeline visualize  strong_baseline_bsb seed=1
    python -m tir_jde.pipeline run        dist_tracker_bsb seed=0 model.detector.epochs=2 tag=smoke
    python -m tir_jde.pipeline compare    strong_baseline_bsb dist_tracker_bsb --seeds 0 1 2
    python -m tir_jde.pipeline make-val   strong_baseline_bsb --fraction 0.15

Each experiment + seed (+ tag) is one MLflow run; every stage logs into it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tir_jde import dataset
from tir_jde.experiment import load_config, resolve_path, run_dir, run_name


# --------------------------------------------------------------------------- stages
def stage_preprocess(cfg: dict) -> dict:
    manifest = dataset.prepare(cfg)
    print(json.dumps({k: manifest[k] for k in ("split_hash", "counts", "root")}, indent=2))
    return manifest


def stage_train(cfg: dict) -> dict:
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

        weights = adapter.train()
        (adapter.run_dir / "weights.json").write_text(json.dumps(weights, indent=2))

        for epoch, metrics in adapter.epoch_metrics(Path(weights.get("detector_results", ""))):
            mlflow.log_metrics({f"train.{k}": v for k, v in metrics.items()}, step=epoch)
        if Path(weights["detector"]).exists():
            mlflow.log_artifact(weights["detector"], artifact_path="weights")
        for log in adapter.logs_dir.glob("train_*.log"):
            mlflow.log_artifact(str(log), artifact_path="logs")
    return weights


def stage_evaluate(cfg: dict) -> dict:
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

        mlflow.log_metrics({f"val.{k}": float(v) for k, v in result["overall"].items()})
        mlflow.log_metric("val.FPS_end_to_end", float(tracked["fps"]))
        mlflow.log_dict(result, "eval/results.json")
        mlflow.log_artifacts(tracked["pred_dir"], artifact_path="eval/predictions")
        (adapter.run_dir / "results.json").write_text(json.dumps(result, indent=2))
        _write_per_sequence_csv(result["per_sequence"], adapter.run_dir / "per_sequence.csv")
        mlflow.log_artifact(str(adapter.run_dir / "per_sequence.csv"), artifact_path="eval")
        for log in adapter.logs_dir.glob("track*.log"):
            mlflow.log_artifact(str(log), artifact_path="logs")
    print(json.dumps(result["overall"], indent=2), f"\nFPS (end-to-end): {tracked['fps']:.1f}")
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
    for name in ("preprocess", "train", "evaluate", "visualize", "run"):
        p = sub.add_parser(name)
        p.add_argument("experiment")
        p.add_argument("overrides", nargs="*")
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
