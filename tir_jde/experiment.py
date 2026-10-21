"""Experiment configuration: composition, overrides and run identity.

An experiment file lists the config parts it is built from::

    defaults: [base, data/bsb, model/strong_baseline]
    name: strong_baseline_bsb

Parts are deep-merged in order, then the experiment's own keys, then
command-line overrides such as ``seed=1`` or ``model.detector.epochs=2``.
String values may reference environment variables as ``${VAR:-default}``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJ_ROOT / "configs"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        expanded = _ENV_PATTERN.sub(repl, value)
        # A value that is exactly one ${...} placeholder is typed (null, numbers, bools);
        # placeholders inside longer strings stay strings.
        if _ENV_PATTERN.fullmatch(value.strip()):
            return yaml.safe_load(expanded) if expanded.strip() else None
        return expanded
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set '{dotted}': '{key}' is not a mapping")
    node[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def parse_overrides(overrides: Iterable[str]) -> list[tuple[str, Any]]:
    parsed = []
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        parsed.append((key.strip(), yaml.safe_load(raw) if raw.strip() else ""))
    return parsed


def load_config(experiment: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Builds the full configuration for one experiment."""
    exp_path = Path(experiment)
    if not exp_path.is_file():
        exp_path = CONFIGS_DIR / "experiment" / f"{experiment}.yaml"
    exp = _read_yaml(exp_path)

    cfg: dict = {}
    for part in exp.pop("defaults", []):
        cfg = deep_merge(cfg, _read_yaml(CONFIGS_DIR / f"{part}.yaml"))
    cfg = deep_merge(cfg, exp)
    cfg.setdefault("name", exp_path.stem)

    for key, value in parse_overrides(overrides):
        set_by_path(cfg, key, value)

    cfg = _expand_env(cfg)
    cfg["paths"] = {k: str(_abs(v)) for k, v in cfg.get("paths", {}).items() if v not in (None, "")}
    return cfg


def _abs(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJ_ROOT / p).resolve()


def resolve_path(path: str | Path) -> Path:
    """Paths in configs are relative to the project root."""
    return _abs(path)


def run_name(cfg: dict) -> str:
    """Stable name of a run: experiment name + seed (+ optional tag)."""
    tag = cfg.get("tag")
    return f"{cfg['name']}-s{cfg.get('seed', 0)}" + (f"-{tag}" if tag else "")


def run_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["work_dir"]) / "runs" / run_name(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}; lists are JSON-encoded."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value)
        else:
            flat[name] = value
    return flat


# Machine-specific values that must not change a run's identity.
_HASH_EXCLUDE = ("paths", "tracking", "model.python")


def config_hash(cfg: dict) -> str:
    """Hash of the settings that define an experiment (not where it runs)."""
    ident = copy.deepcopy(cfg)
    for dotted in _HASH_EXCLUDE:
        *parents, last = dotted.split(".")
        node = get_by_path(ident, ".".join(parents)) if parents else ident
        if isinstance(node, dict):
            node.pop(last, None)
    blob = json.dumps(ident, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save_config(cfg: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
